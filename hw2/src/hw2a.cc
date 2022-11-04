#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <iostream>

// Global variables
int* image;
int thread_num = 0;
int iters;
int width, height;
int curr_x_start = 0, curr_y_start = 0;
int calc_width = 100, calc_height = 100;
double left, right, lower, upper;
double x_step, y_step;

void calc_mandelbrot_set(int x_start, int x_end, int y_start, int y_end) {
    for(int j = y_start; j < y_end; ++j) {
        double y0 = j * y_step + lower;
        for(int i = x_start; i < x_end; ++i) {
            double x0 = i * x_step + left;

            int repeats = 0;
            double x = 0;
            double y = 0;
            double length_squared = 0;
            while(repeats < iters && length_squared < 4) {
                double temp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = temp;
                length_squared = x * x + y * y;
                ++repeats;
            }
            image[j * width + i] = repeats;
        }
    }
}

void* thread_func(void* thread_id) {
    while(true) {
        // Declare variables
        int x_start, x_end, y_start, y_end;

        // Entry section
        pthread_mutex_t mutex;
        pthread_mutex_init(&mutex, NULL);
        pthread_mutex_lock(&mutex);

        // Critical section
        if(curr_y_start >= height) {
            pthread_mutex_unlock(&mutex);
            pthread_mutex_destroy(&mutex);
            break;
        }

        x_start = curr_x_start;
        x_end = x_start + calc_width;
        y_start = curr_y_start;
        y_end = y_start + calc_height;

        if(x_end > width) {
            x_end = width;
        }
        if(y_end > height) {
            y_end = height;
        }

        curr_x_start = x_end;
        if(curr_x_start >= width) {
            curr_x_start = 0;
            curr_y_start = y_end;
        }

        // Exit section
        pthread_mutex_unlock(&mutex);
        pthread_mutex_destroy(&mutex);

        // Calculate
        calc_mandelbrot_set(x_start, x_end, y_start, y_end);
    }

    // Exit
    pthread_exit(NULL);
}

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
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
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
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

int main(int argc, char** argv) {
    // Detect how many CPUs are available
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);

    // Argument parsing
    assert(argc == 9);
    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);

    x_step = (right - left) / width;
    y_step = (upper - lower) / height;

    // Allocate memory for image
    image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    // Create threads
    thread_num = CPU_COUNT(&cpu_set);
    pthread_t threads[thread_num];
    int thread_id[thread_num];
    for(int i = 0; i < thread_num; ++i) {
        thread_id[i] = i;
        pthread_create(&threads[i], NULL, thread_func, (void*)&thread_id[i]);
    }

    // Join threads
    for(int i = 0; i < thread_num; ++i) {
        pthread_join(threads[i], NULL);
    }

    // Draw and cleanup
    write_png(filename, iters, width, height, image);
    free(image);
}
