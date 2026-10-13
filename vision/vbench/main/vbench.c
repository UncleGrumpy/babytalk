// vbench: MMRT's 2-D ops on the board. At boot it runs the test models flashed to the "model"
// partition (vision/synth.py: random weights, real shapes) and prints one "@@R {json}" line per
// configuration -- milliseconds per image, per-op-kind time, per-op time, and the FNV-1a hash of
// the output, which must equal the "out_fnv" synth.py printed on the PC for the same model.
//
//   python3 vision/synth.py --check
//   vision/vbench/idf.sh esp32s3 build flash      (then the models: vision/vbench/README.md)
//
// Reset the board to run it again.
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "esp_partition.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "mmrt.h"
#include "mmrt_ref.h"
#ifdef MMRT_S3
#include "mmrt_s3.h"
#else
#include "mmrt_port.h"
#endif

#ifdef MMRT_S3
#define CHIP "esp32s3"
#else
#define CHIP "esp32p4"
#endif

static const struct {
    const char *name;
    uint32_t off;
} MODELS[] = {{"mobilenetv2_224", 0}, {"mobilefacenet_112", 0x400000}};

// ---------------------------------------------------------------- memory
static size_t g_sram_max = 160 * 1024, g_sram_reserve = 48 * 1024;  // activations this small go internal
static void *psram_alloc(size_t n) { return heap_caps_aligned_alloc(16, n ? n : 16, MALLOC_CAP_SPIRAM); }
static void psram_free(void *p) { heap_caps_free(p); }
static void *act_alloc(size_t n)
{
    if (n && n <= g_sram_max && heap_caps_get_free_size(MALLOC_CAP_INTERNAL) >= n + g_sram_reserve) {
        void *p = heap_caps_aligned_alloc(16, n, MALLOC_CAP_INTERNAL);
        if (p) return p;
    }
    return psram_alloc(n);
}

// ---------------------------------------------------------------- per-op timing
static int64_t g_last_us, g_kind_us[9];
static int32_t *g_op_us;
static void on_op(int i, const mmrt_op_t *op, const int8_t *y, int T, int C)
{
    (void)y, (void)T, (void)C;
    int64_t now = esp_timer_get_time();
    if (op->kind < 9) g_kind_us[op->kind] += now - g_last_us;
    if (g_op_us) g_op_us[i] = (int32_t)(now - g_last_us);
    g_last_us = now;
}

static uint32_t fnv1a(const int8_t *p, size_t n)
{
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++) h = (h ^ (uint8_t)p[i]) * 16777619u;
    return h;
}

// the input synth.py computes too (lcg_bytes): the top byte of an LCG
static void vbench_input(int8_t *x, size_t n)
{
    uint32_t s = 12345;
    for (size_t i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        x[i] = (int8_t)(s >> 24);
    }
}

// ---------------------------------------------------------------- the 2-D depthwise kernel vs the reference
#ifdef MMRT_S3
static int dw2d_test(void)
{
    struct { int H, W, C, K, stride, pad, shift, relu; } cases[] = {
        {9, 11, 32, 3, 1, 1, 6, 1}, {9, 11, 32, 3, 2, 1, 5, 0}, {8, 8, 48, 3, 2, 1, 0, 1}, {7, 7, 64, 7, 1, 0, 8, 0},
        {5, 6, 16, 5, 1, 2, 7, 1},  {6, 5, 16, 1, 1, 0, 3, 0},  {112, 112, 32, 3, 1, 1, 7, 1}, {13, 17, 96, 3, 2, 0, 9, 1},
    };
    uint32_t r = 99;
    int fails = 0;
    for (size_t k = 0; k < sizeof(cases) / sizeof(cases[0]); k++) {
        const int H = cases[k].H, W = cases[k].W, C = cases[k].C, K = cases[k].K, s = cases[k].stride, p = cases[k].pad;
        const int Ho = (H + 2 * p - K) / s + 1, Wo = (W + 2 * p - K) / s + 1;
        int8_t *x = psram_alloc((size_t)H * W * C), *w = psram_alloc((size_t)C * K * K);
        int8_t *y0 = psram_alloc((size_t)Ho * Wo * C), *y1 = psram_alloc((size_t)Ho * Wo * C);
        for (int i = 0; i < H * W * C; i++) x[i] = (int8_t)((r = r * 1664525u + 1013904223u) >> 24);
        for (int i = 0; i < C * K * K; i++) w[i] = (int8_t)((r = r * 1664525u + 1013904223u) >> 24);
        int64_t t0 = esp_timer_get_time();
        mmrt_dwconv2d_ref(x, H, W, C, w, K, s, p, cases[k].shift, cases[k].relu, y0, Ho, Wo);
        int64_t t1 = esp_timer_get_time();
        mmrt_s3_dwconv2d(x, H, W, C, w, K, s, p, cases[k].shift, cases[k].relu, y1, Ho, Wo);
        int64_t t2 = esp_timer_get_time();
        int bad = 0;
        for (int i = 0; i < Ho * Wo * C; i++) bad += y0[i] != y1[i];
        fails += bad > 0;
        printf("@@R {\"test\":\"dwconv2d\",\"H\":%d,\"W\":%d,\"C\":%d,\"K\":%d,\"stride\":%d,\"pad\":%d,\"bad\":%d,"
               "\"ref_us\":%lld,\"s3_us\":%lld,\"mmac_s\":%.0f}\n", H, W, C, K, s, p, bad, (long long)(t1 - t0),
               (long long)(t2 - t1), (double)Ho * Wo * C * K * K / (double)(t2 - t1 > 0 ? t2 - t1 : 1));
        psram_free(x), psram_free(w), psram_free(y0), psram_free(y1);
    }
    return fails;
}
#endif

// ---------------------------------------------------------------- one model, one configuration
static void run_config(mmrt_model_t *m, const char *name, const int8_t *x, int T, const char *weights, int iters)
{
    const int n = (int)m->hdr->n_ops;
    g_op_us = calloc(n, sizeof(int32_t));
    memset(g_kind_us, 0, sizeof(g_kind_us));
    mmrt_trace = on_op;
    int T_out = 0;
    g_last_us = esp_timer_get_time();
    int64_t t0 = g_last_us;
    const int8_t *y = mmrt_run(m, x, T, &T_out, act_alloc, psram_free);  // the first run: profiled, untimed
    int64_t first_us = esp_timer_get_time() - t0;
    mmrt_trace = NULL;
    if (!y) {
        printf("@@R {\"model\":\"%s\",\"error\":\"mmrt_run failed (out of memory?)\"}\n", name);
        free(g_op_us);
        g_op_us = NULL;
        return;
    }
    const int C_out = m->tensors[m->hdr->output].channels;
    const uint32_t h = fnv1a(y, (size_t)T_out * C_out);
    int64_t best = INT64_MAX, sum = 0;
    for (int i = 0; i < iters; i++) {
        t0 = esp_timer_get_time();
        y = mmrt_run(m, x, T, &T_out, act_alloc, psram_free);
        int64_t dt = esp_timer_get_time() - t0;
        if (!y || fnv1a(y, (size_t)T_out * C_out) != h) {
            printf("@@R {\"model\":\"%s\",\"error\":\"run %d differs from the first\"}\n", name, i);
            break;
        }
        sum += dt;
        if (dt < best) best = dt;
    }
#ifdef MMRT_S3
    const int cores = mmrt_s3_cores;
#else
    const int cores = 2;
#endif
    printf("@@R {\"chip\":\"" CHIP "\",\"model\":\"%s\",\"weights\":\"%s\",\"cores\":%d,\"ms_min\":%.1f,\"ms_mean\":%.1f,"
           "\"ms_first\":%.1f,\"fnv\":\"%08" PRIx32 "\",\"ms_by_kind\":{\"conv1x1\":%.1f,\"dwconv2d\":%.1f,\"im2col\":%.1f,"
           "\"add\":%.1f,\"mean\":%.1f},\"psram_free\":%u,\"internal_free\":%u}\n",
           name, weights, cores, best / 1000.0, iters ? sum / 1000.0 / iters : 0.0, first_us / 1000.0, h,
           g_kind_us[MMRT_CONV1X1] / 1000.0, g_kind_us[MMRT_DWCONV2D] / 1000.0, g_kind_us[MMRT_IM2COL] / 1000.0,
           g_kind_us[MMRT_ADD] / 1000.0, g_kind_us[MMRT_MEAN] / 1000.0,
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM), (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
    printf("@@OPS %s %s", name, weights);
    for (int i = 0; i < n; i++) printf(" %d", (int)g_op_us[i]);
    printf("\n");
    free(g_op_us);
    g_op_us = NULL;
}

static void run_model(const esp_partition_t *part, const void *base, int idx, int iters)
{
    const char *name = MODELS[idx].name;
    const uint8_t *img = (const uint8_t *)base + MODELS[idx].off;
    const size_t avail = part->size - MODELS[idx].off;
    mmrt_model_t m;
    int rc = mmrt_open(&m, img, avail);
    if (rc) {
        printf("@@R {\"model\":\"%s\",\"error\":\"mmrt_open %d: flash vision/out/%s.mmrt at 0x%" PRIx32 "\"}\n", name, rc,
               name, (uint32_t)(part->address + MODELS[idx].off));
        return;
    }
    // the input: a square image, its width the first 2-D op's
    int W = 0;
    for (uint32_t i = 0; i < m.hdr->n_ops && !W; i++) W = m.ops[i].W;
    const int T = W * W, C = m.tensors[m.hdr->input].channels;
    int8_t *x = psram_alloc((size_t)T * C);
    vbench_input(x, (size_t)T * C);
    printf("@@R {\"model\":\"%s\",\"ops\":%" PRIu32 ",\"blob_mb\":%.2f,\"input\":[%d,%d,%d]}\n", name, m.hdr->n_ops,
           m.hdr->blob_size / 1e6, W, W, C);

    run_config(&m, name, x, T, "flash", iters);
    size_t cached = mmrt_cache_weights(&m, 12u << 20, psram_alloc, psram_free);
    if (cached) run_config(&m, name, x, T, "psram", iters);
#ifdef MMRT_S3
    mmrt_s3_cores = 1;
    run_config(&m, name, x, T, cached ? "psram" : "flash", iters);
    mmrt_s3_cores = 2;
#endif
    mmrt_close(&m, psram_free);
    psram_free(x);
}

void app_main(void)
{
    vTaskDelay(pdMS_TO_TICKS(1500));  // time for a serial monitor to attach
    printf("@@START vbench " CHIP "\n");
#ifdef MMRT_S3
    mmrt_s3_init();
    int fails = dw2d_test();
    printf("@@R {\"test\":\"dwconv2d\",\"fails\":%d}\n", fails);
#endif
    const esp_partition_t *p = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "model");
    const void *base;
    esp_partition_mmap_handle_t h;
    if (!p || esp_partition_mmap(p, 0, p->size, ESP_PARTITION_MMAP_DATA, &base, &h) != ESP_OK) {
        printf("@@R {\"error\":\"no model partition to map\"}\n");
    } else {
        for (int i = 0; i < (int)(sizeof(MODELS) / sizeof(MODELS[0])); i++) run_model(p, base, i, 5);
    }
    printf("@@DONE\n");
    for (;;) vTaskDelay(pdMS_TO_TICKS(1000));
}
