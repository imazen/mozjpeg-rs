/* Library-level mozjpeg oracle for mozjpeg-rs parity tests.
 *
 * Primary role: produce `optimize_coding = FALSE` with trellis enabled,
 * which no cjpeg flag combination yields (`cjpeg -baseline` still runs the
 * JCP_MAX_COMPRESSION defaults with optimize_coding=TRUE, and `-revert`
 * selects JCP_FASTEST which disables trellis entirely). This driver goes
 * through the library API directly: jpeg_set_defaults() under the default
 * JCP_MAX_COMPRESSION profile, then clears the progressive scan script and
 * forces optimize_coding=FALSE — unless flags say otherwise.
 *
 * It also exposes parameters cjpeg cannot express:
 *   -optimize       optimize_coding = TRUE (default is FALSE)
 *   -progressive    keep the progressive scan script from the defaults
 *                   (with -optimize, the master's scan search still runs)
 *   -eob            trellis_eob_opt = TRUE (no cjpeg flag exists)
 *
 * Input:  P6 (RGB) or P5 (grayscale) Netpbm on stdin
 * Output: JPEG on stdout
 *
 * Usage: trellis_noopt_oracle -quality N [-sample HxV] [-restart N]
 *            [-optimize] [-progressive] [-eob]
 *
 * Build (from the mozjpeg-rs repo root):
 *   cc -O2 -I ../mozjpeg -I ../mozjpeg/build \
 *      tests/oracle/trellis_noopt_oracle.c ../mozjpeg/build/libjpeg.a \
 *      -o <out>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <jpeglib.h>
#include <jpegint.h>
#include <setjmp.h>

static void usage(void) {
    fprintf(stderr, "usage: trellis_noopt_oracle -quality N [-sample HxV] [-restart N]\n"
                    "            [-optimize] [-progressive] [-eob]\n");
    exit(1);
}

int main(int argc, char **argv) {
    int quality = 75;
    int hsamp = 2, vsamp = 2;
    int restart = 0;
    int optimize = 0;
    int progressive = 0;
    int eob = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-quality") && i + 1 < argc)
            quality = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-sample") && i + 1 < argc)
            sscanf(argv[++i], "%dx%d", &hsamp, &vsamp);
        else if (!strcmp(argv[i], "-restart") && i + 1 < argc)
            restart = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-optimize"))
            optimize = 1;
        else if (!strcmp(argv[i], "-progressive"))
            progressive = 1;
        else if (!strcmp(argv[i], "-eob"))
            eob = 1;
        else
            usage();
    }

    /* Parse Netpbm header: P5 or P6, whitespace/comments, maxval 255. */
    int magic = getchar();
    int kind = getchar();
    if (magic != 'P' || (kind != '6' && kind != '5')) {
        fprintf(stderr, "expected P5/P6 input\n");
        return 1;
    }
    int cpp = (kind == '6') ? 3 : 1;
    long w = 0, h = 0, maxval = 0;
    for (int field = 0; field < 3;) {
        int c = getchar();
        if (c == '#') {
            while ((c = getchar()) != '\n' && c != EOF)
                ;
            continue;
        }
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n')
            continue;
        ungetc(c, stdin);
        long v = 0;
        while ((c = getchar()) >= '0' && c <= '9')
            v = v * 10 + (c - '0');
        if (field == 0) w = v;
        else if (field == 1) h = v;
        else maxval = v;
        field++;
    }
    if (w <= 0 || h <= 0 || maxval != 255) {
        fprintf(stderr, "bad header\n");
        return 1;
    }

    size_t npix = (size_t)w * (size_t)h * (size_t)cpp;
    unsigned char *pixels = (unsigned char *)malloc(npix);
    if (!pixels || fread(pixels, 1, npix, stdin) != npix) {
        fprintf(stderr, "short input\n");
        return 1;
    }

    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, stdout);

    cinfo.image_width = (JDIMENSION)w;
    cinfo.image_height = (JDIMENSION)h;
    cinfo.input_components = cpp;
    cinfo.in_color_space = (cpp == 3) ? JCS_RGB : JCS_GRAYSCALE;

    jpeg_set_defaults(&cinfo);

    /* Sequential baseline unless -progressive: drop the script installed by
     * the JCP_MAX_COMPRESSION defaults. validate_script() with scan_info ==
     * NULL produces the default single sequential scan; progressive_mode
     * stays FALSE, so jcmaster does not re-force optimize_coding. With
     * -progressive the defaults' script stays and the master's optimize_scans
     * search runs exactly as it does under cjpeg. */
    if (!progressive) {
        cinfo.scan_info = NULL;
        cinfo.num_scans = 0;
    }
    cinfo.optimize_coding = optimize;
    cinfo.master->trellis_quant = TRUE;
    cinfo.master->trellis_quant_dc = TRUE;
    if (eob)
        cinfo.master->trellis_eob_opt = TRUE;

    jpeg_set_quality(&cinfo, quality, TRUE); /* force_baseline = TRUE */

    if (cpp == 3) {
        cinfo.comp_info[0].h_samp_factor = hsamp;
        cinfo.comp_info[0].v_samp_factor = vsamp;
        cinfo.comp_info[1].h_samp_factor = cinfo.comp_info[1].v_samp_factor = 1;
        cinfo.comp_info[2].h_samp_factor = cinfo.comp_info[2].v_samp_factor = 1;
    }
    cinfo.restart_interval = restart;

    jpeg_start_compress(&cinfo, TRUE);
    size_t stride = (size_t)w * (size_t)cpp;
    while (cinfo.next_scanline < cinfo.image_height) {
        JSAMPROW row = pixels + (size_t)cinfo.next_scanline * stride;
        jpeg_write_scanlines(&cinfo, &row, 1);
    }
    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    free(pixels);
    return 0;
}
