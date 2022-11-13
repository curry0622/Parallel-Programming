/* without vectorization */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <assert.h>
#include <math.h>
#include <png.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <emmintrin.h>
#include <smmintrin.h>
#include <pmmintrin.h>

// Global variables
int rank, size;
char *filename;
int iters;
double left, right, lower, upper;
int width, height;
double x_step, y_step;
int calc_height;
int *buffer, *image;

// Arguments parsing
void parse_args(int argc, char **argv) {
    assert(argc == 9);

    filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);

    x_step = (right - left) / width;
    y_step = (upper - lower) / height;
    calc_height = ceil((double)height / size);

    buffer = (int *)malloc(calc_height * width * sizeof(int));
    image = (int *)malloc(size * calc_height * width * sizeof(int));
}

// Calculate the length_squared
void calc_lsqr(double* x, double* y, double* x0, double* y0, double* lsqr) {
    double temp = (*x) * (*x) - (*y) * (*y) + (*x0);
    *y = 2 * (*x) * (*y) + (*y0);
    *x = temp;
    *lsqr = (*x) * (*x) + (*y) * (*y);
}

// Calculate the length_squared using SSE
void calc_lsqr_sse(__m128d* x, __m128d* y, __m128d* x0, __m128d* y0, __m128d* lsqr) {
    __m128d temp = _mm_add_pd(_mm_sub_pd(_mm_mul_pd(*x, *x), _mm_mul_pd(*y, *y)), *x0);
    *y = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(*x, *y), _mm_set1_pd(2)), *y0);
    *x = temp;
    *lsqr = _mm_add_pd(_mm_mul_pd(*x, *x), _mm_mul_pd(*y, *y));
}

// Calculate the mandelbrot set using omp
void calc_mandelbrot_set_omp() {
    __m128d zero = _mm_setzero_pd(),  four = _mm_set_pd1(4);

    for(int j = rank, row = 0; j < height; j += size, row++) {
        double y0 = j * y_step + lower;
        __m128d y00 = _mm_load1_pd(&y0);
        int end = (width >> 1) << 1;

#pragma omp parallel for schedule(dynamic, 1)
        for(int i = 0; i < end; i += 2) {
            // Initialization
            double x0[2] = {i * x_step + left, (i + 1) * x_step + left};
            __m128d x00 = _mm_load_pd(x0);
            __m128d x = zero, y = zero, lsqr = zero;
            int repeats[2] = {0, 0};
            bool finish[2] = {false, false};

            // Calculate the number of iterations
            while(!finish[0] || !finish[1]) {
                if(!finish[0]) {
                    if(repeats[0] < iters && _mm_comilt_sd(lsqr, four))
                        repeats[0]++;
                    else
                        finish[0] = true;
                }
                if(!finish[1]) {
                    if(repeats[1] < iters && _mm_comilt_sd(_mm_unpackhi_pd(lsqr, lsqr), four))
                        repeats[1]++;
                    else
                        finish[1] = true;
                }
                calc_lsqr_sse(&x, &y, &x00, &y00, &lsqr);
            }

            // Write buffer
            buffer[row * width + i] = repeats[0];
            buffer[row * width + i + 1] = repeats[1];
        }

        // Remainder
        if(end < width) {
            double x = 0, y = 0, lsqr = 0, x0 = end * x_step + left;
            int repeats = 0;
            while(repeats < iters && lsqr < 4) {
                calc_lsqr(&x, &y, &x0, &y0, &lsqr);
                repeats++;
            }
            buffer[row * width + end] = repeats;
        }
    }
}

// Write image
void write_png() {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for(int y = height - 1; y >= 0; y--) {
        memset(row, 0, row_size);
        // Calculate the base index
        int base = y % size * calc_height + y / size;
        for(int x = 0; x < width; x++) {
            int p = image[base * width + x];
            png_bytep color = row + x * 3;
            if(p != iters) {
                if(p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = (p & 15) << 4;
                } else
                    color[0] = (p & 15) << 4;
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int main(int argc, char **argv) {
    // MPI init
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Parsing arguments
    parse_args(argc, argv);

    // calculate mandelbrot set using omp
    calc_mandelbrot_set_omp();

    // MPI gather result
    MPI_Gather(buffer, calc_height * width, MPI_INT, image, calc_height * width, MPI_INT, 0, MPI_COMM_WORLD);

    // Draw image and cleanup
    if (rank == 0)
        write_png();
    free(buffer), free(image);
    MPI_Finalize();
    return 0;
}