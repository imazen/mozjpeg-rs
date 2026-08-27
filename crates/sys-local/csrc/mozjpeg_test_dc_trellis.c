/*
 * mozjpeg_test_dc_trellis.c
 *
 * Test export for the DC trellis optimization, for FFI validation of the
 * Rust `mozjpeg_rs::trellis::dc_trellis_optimize` against the C algorithm.
 *
 * This is a cinfo-free reimplementation of the DC-coefficient branch of
 * `quantize_trellis()` in mozjpeg's jcdctmgr.c (the `trellis_quant_dc`
 * sections: candidate generation, DPCM dynamic programming, and backtrack),
 * in the same style as the other exports in mozjpeg_test_exports.c.
 *
 * It lives in mozjpeg-rs (compiled by crates/sys-local/build.rs) rather than
 * in the mozjpeg C tree because the symbol was declared on the Rust side
 * without a matching C definition, which left `cargo test -p sys-local`
 * unlinkable against every revision of the mozjpeg source.
 *
 * Deliberately header-free: it uses <stdint.h> equivalents of mozjpeg's
 * JCOEF (short) and UINT16 (unsigned short) so it does not depend on the
 * CMake-generated jconfig.h. Lambda is computed with powf() exactly as
 * mozjpeg_test_trellis_quantize_block() does, so the two exports agree.
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

/* jcdctmgr.c: DC_TRELLIS_MAX_CANDIDATES */
#define DC_TRELLIS_MAX_CANDIDATES 9

/* jcdctmgr.c: get_num_dc_trellis_candidates() */
static int get_num_dc_trellis_candidates(int dc_quantval)
{
    int n = (2 + 60 / dc_quantval) | 1; /* force odd */
    return n < DC_TRELLIS_MAX_CANDIDATES ? n : DC_TRELLIS_MAX_CANDIDATES;
}

/* jpeg_nbits.h: JPEG_NBITS(x) for x >= 0 (0 -> 0, else bit length) */
static int dc_nbits(int value)
{
    int bits = 0;
    if (value < 0)
        value = -value;
    while (value) {
        bits++;
        value >>= 1;
    }
    return bits;
}

/* jcdctmgr.c: compute_dc_huffman_bits() */
static float compute_dc_huffman_bits(int dc_delta, const signed char *dc_huffsi)
{
    int nbits = dc_nbits(dc_delta);
    return (float)(nbits + dc_huffsi[nbits]);
}

/*
 * DC trellis optimization on a sequence of blocks.
 *
 * raw_dc          - Raw DC coefficients (num_blocks values, each scaled by 8)
 * ac_norms        - AC energy per block (num_blocks values, sum(ac^2)/63)
 * quantized_dc    - Output optimized DC coefficients (num_blocks values)
 * num_blocks      - Number of blocks in the chain
 * dc_quantval     - DC quantization table value
 * dc_huffsi       - DC Huffman code sizes (17 values, for categories 0-16)
 * last_dc         - DC predictor for the first block (DPCM)
 * lambda_log_scale1 / lambda_log_scale2 - lambda parameters (14.75 / 16.5)
 */
void mozjpeg_test_dc_trellis_optimize(
    const int *raw_dc,
    const float *ac_norms,
    int16_t *quantized_dc,
    int num_blocks,
    uint16_t dc_quantval,
    const signed char *dc_huffsi,
    int16_t last_dc,
    float lambda_log_scale1,
    float lambda_log_scale2)
{
    const int max_coef_bits = 8 + 2; /* data_precision + 2 */
    const int max_coef_value = (1 << max_coef_bits) - 1; /* 1023 */
    const int dc_trellis_candidates = get_num_dc_trellis_candidates(dc_quantval);
    const int q = 8 * dc_quantval;
    /* init_lambda_table() mode 1: flat weights 1/q^2 */
    const float lambda_dc_weight = 1.0f / (dc_quantval * dc_quantval);
    float *accumulated_dc_cost[DC_TRELLIS_MAX_CANDIDATES];
    int *dc_cost_backtrack[DC_TRELLIS_MAX_CANDIDATES];
    int16_t *dc_candidate[DC_TRELLIS_MAX_CANDIDATES];
    int i, k, l, bi, j;

    if (num_blocks <= 0)
        return;

    for (i = 0; i < dc_trellis_candidates; i++) {
        accumulated_dc_cost[i] = (float *)malloc((size_t)num_blocks * sizeof(float));
        dc_cost_backtrack[i] = (int *)malloc((size_t)num_blocks * sizeof(int));
        dc_candidate[i] = (int16_t *)malloc((size_t)num_blocks * sizeof(int16_t));
        if (!accumulated_dc_cost[i] || !dc_cost_backtrack[i] || !dc_candidate[i])
            abort();
    }

    for (bi = 0; bi < num_blocks; bi++) {
        /* compute_block_lambda() with lambda_base = 1 (mode 1) */
        float norm = ac_norms[bi];
        float lambda;
        float lambda_dc;
        int sign = raw_dc[bi] >> 31;
        int x = abs(raw_dc[bi]);
        int qval = (x + q / 2) / q;

        if (lambda_log_scale2 > 0.0f) {
            lambda = powf(2.0f, lambda_log_scale1) /
                     (powf(2.0f, lambda_log_scale2) + norm);
        } else {
            lambda = powf(2.0f, lambda_log_scale1 - 12.0f);
        }
        lambda_dc = lambda * lambda_dc_weight;

        for (k = 0; k < dc_trellis_candidates; k++) {
            int delta;
            int dc_delta;
            float dc_candidate_dist;
            float cost;
            int cand = qval - dc_trellis_candidates / 2 + k;

            if (cand > max_coef_value)
                cand = max_coef_value;
            if (cand < -max_coef_value)
                cand = -max_coef_value;

            delta = cand * q - x;
            dc_candidate_dist = delta * delta * lambda_dc;
            /* apply sign: sign is 0 or -1 */
            dc_candidate[k][bi] = (int16_t)(cand * (1 + 2 * sign));

            if (bi == 0) {
                dc_delta = dc_candidate[k][bi] - last_dc;
                cost = compute_dc_huffman_bits(dc_delta, dc_huffsi) + dc_candidate_dist;
                accumulated_dc_cost[k][0] = cost;
                dc_cost_backtrack[k][0] = -1;
            } else {
                for (l = 0; l < dc_trellis_candidates; l++) {
                    dc_delta = dc_candidate[k][bi] - dc_candidate[l][bi - 1];
                    cost = compute_dc_huffman_bits(dc_delta, dc_huffsi) +
                           dc_candidate_dist + accumulated_dc_cost[l][bi - 1];
                    if (l == 0 || cost < accumulated_dc_cost[k][bi]) {
                        accumulated_dc_cost[k][bi] = cost;
                        dc_cost_backtrack[k][bi] = l;
                    }
                }
            }
        }
    }

    /* Pick the cheapest final state and backtrack */
    j = 0;
    for (i = 1; i < dc_trellis_candidates; i++) {
        if (accumulated_dc_cost[i][num_blocks - 1] < accumulated_dc_cost[j][num_blocks - 1])
            j = i;
    }
    for (bi = num_blocks - 1; bi >= 0; bi--) {
        quantized_dc[bi] = dc_candidate[j][bi];
        j = dc_cost_backtrack[j][bi];
    }

    for (i = 0; i < dc_trellis_candidates; i++) {
        free(accumulated_dc_cost[i]);
        free(dc_cost_backtrack[i]);
        free(dc_candidate[i]);
    }
}
