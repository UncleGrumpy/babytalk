// mmrt float32 ops (mmrt_f32.h).
#include "mmrt_f32.h"

#include <math.h>
#include <stddef.h>

void mmrt_f32_linear(const float *W, const float *b, const float *x, int T, int N, int C, float *y)
{
    for (int n = 0; n < N; n++) {
        const float *w = W + (size_t)n * C;
        const float bn = b ? b[n] : 0.0f;
        int t = 0;
        for (; t + 4 <= T; t += 4) {   // 4 rows per weight load, 4 independent sums (keeps the FPU busy)
            const float *x0 = x + (size_t)t * C, *x1 = x0 + C, *x2 = x1 + C, *x3 = x2 + C;
            float a0 = bn, a1 = bn, a2 = bn, a3 = bn;
            for (int c = 0; c < C; c++) {
                const float wc = w[c];
                a0 += wc * x0[c];
                a1 += wc * x1[c];
                a2 += wc * x2[c];
                a3 += wc * x3[c];
            }
            y[(size_t)t * N + n] = a0;
            y[(size_t)(t + 1) * N + n] = a1;
            y[(size_t)(t + 2) * N + n] = a2;
            y[(size_t)(t + 3) * N + n] = a3;
        }
        for (; t < T; t++) {
            const float *xt = x + (size_t)t * C;
            float acc = bn;
            for (int c = 0; c < C; c++) acc += w[c] * xt[c];
            y[(size_t)t * N + n] = acc;
        }
    }
}

void mmrt_f32_layernorm(const float *x, const float *w, const float *b, int T, int C, float eps, float *y)
{
    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * C;
        float *yt = y + (size_t)t * C;
        float mu = 0, var = 0;
        for (int i = 0; i < C; i++) mu += xt[i];
        mu /= C;
        for (int i = 0; i < C; i++) var += (xt[i] - mu) * (xt[i] - mu);
        const float inv = 1.0f / sqrtf(var / C + eps);
        for (int i = 0; i < C; i++) yt[i] = (xt[i] - mu) * inv * w[i] + b[i];
    }
}

void mmrt_f32_softmax(float *x, int n)
{
    float mx = x[0], s = 0;
    for (int i = 1; i < n; i++) mx = x[i] > mx ? x[i] : mx;
    for (int i = 0; i < n; i++) s += (x[i] = expf(x[i] - mx));
    const float inv = 1.0f / s;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

void mmrt_f32_relu(float *x, int n)
{
    for (int i = 0; i < n; i++) x[i] = x[i] > 0 ? x[i] : 0;
}

void mmrt_f32_attention(const float *qkv, int T, int D, int heads, float *out)
{
    const int hd = D / heads, S = 3 * D;
    const float scale = 1.0f / sqrtf((float)hd);
    float a[256];
    for (int h = 0; h < heads; h++)
        for (int t = 0; t < T; t++) {
            const float *q = qkv + (size_t)t * S + h * hd;
            for (int s = 0; s < T; s++) {
                const float *k = qkv + (size_t)s * S + D + h * hd;
                float acc = 0;
                for (int i = 0; i < hd; i++) acc += q[i] * k[i];
                a[s] = acc * scale;
            }
            mmrt_f32_softmax(a, T);
            float *o = out + (size_t)t * D + h * hd;
            for (int i = 0; i < hd; i++) o[i] = 0;
            for (int s = 0; s < T; s++) {
                const float *v = qkv + (size_t)s * S + 2 * D + h * hd;
                for (int i = 0; i < hd; i++) o[i] += a[s] * v[i];
            }
        }
}
