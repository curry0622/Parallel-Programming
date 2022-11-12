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

void write_png(const char* filename, int iters, int width, int height, int size, int calc_height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for(int y = height - 1; y >= 0; y--) {
        memset(row, 0, row_size);
        for(int x = 0; x < width; x++) {
            int p = buffer[(y % size * calc_height + y / size) * width + x];
            png_bytep color = row + x * 3;
            if(p != iters) {
                if(p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = (p & 15) << 4;
                } else {
                    color[0] = (p & 15) << 4;
                }
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
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Argument parsing
    assert(argc == 9);
    char *filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);

    double x_step = (right - left) / width;
    double y_step = (upper - lower) / height;
    int calc_height = ceil((double)height / size);

    // Memory allocation
    int *buffer = (int *)malloc(calc_height * width * sizeof(int));
    int *image = (int *)malloc(size * calc_height * width * sizeof(int));

    // Mandelbrot set
    // int j = height - 1 - rank, row = 0; j >= 0; j -= size, row++
    // int j = 0, row = 0; j < height; j += size, row++
    for(int j = rank, row = 0; j < height; j += size, row++) {
        double y0 = j * y_step + lower;
#pragma omp parallel for schedule(dynamic, 1)
        for(int i = 0; i < width; ++i) {
            // Initialization
            int repeats = 0;
            double x = 0, y = 0, lsqr = 0, x0 = i * x_step + left;

            // Calculate the number of iterations
            while(repeats < iters && lsqr < 4) {
                double temp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = temp;
                lsqr = x * x + y * y;
                repeats++;
            }

            // Write buffer
            buffer[row * width + i] = repeats;
        }
    }

    // MPI gather result
    MPI_Gather(buffer, calc_height * width, MPI_INT, image, calc_height * width, MPI_INT, 0, MPI_COMM_WORLD);

    // Draw image and cleanup
    if (rank == 0)
        write_png(filename, iters, width, height, size, calc_height, image);
    free(buffer);
    free(image);
    MPI_Finalize();
    return 0;
}