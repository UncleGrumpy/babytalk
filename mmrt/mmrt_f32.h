// mmrt float32 ops: the small dense layers that sit around an int8 MMRT graph (a classifier
// head, a pooling step, a few attention layers over a handful of tokens), callable from your
// own code. Plain C, any chip; no allocation, no state.
//
// Layout conventions (PyTorch's, so exported weights need no transposing):
//   activations   row-major [T][C]: T rows (tokens, frames) of C values
//   weights       [N][C] as nn.Linear.weight: output n's C weights contiguous
//
// Weights may live in memory-mapped flash. mmrt_f32_linear reads each weight row once and
// applies it to every input row, so flash is read once per call rather than once per row
// (on an ESP32-S3, reading per row made a 2-layer attention mixer over 21 tokens 3.5x slower).
//
// API version: MMRT_API_VERSION in mmrt.h. These signatures change only with it.
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// y[t][n] = b[n] + sum_c W[n][c] * x[t][c]; b may be NULL. x and y must not overlap.
void mmrt_f32_linear(const float *W, const float *b, const float *x, int T, int N, int C, float *y);

// Per row: y = (x - mean) / sqrt(var + eps) * w + b (biased variance, as nn.LayerNorm).
// x and y may be the same buffer.
void mmrt_f32_layernorm(const float *x, const float *w, const float *b, int T, int C, float eps, float *y);

// In place: x = softmax(x), numerically stable.
void mmrt_f32_softmax(float *x, int n);

// In place: x = max(x, 0).
void mmrt_f32_relu(float *x, int n);

// Multi-head self-attention core over T tokens, every token attending to every token:
// qkv [T][3D] holds each token's query, key and value (nn.MultiheadAttention's in_proj
// output, heads side by side within each), out [T][D] gets softmax(q k^T / sqrt(D/heads)) v
// per head, heads side by side. The output projection is a separate mmrt_f32_linear.
// D/heads at most 256; T at most 256.
void mmrt_f32_attention(const float *qkv, int T, int D, int heads, float *out);

#ifdef __cplusplus
}
#endif
