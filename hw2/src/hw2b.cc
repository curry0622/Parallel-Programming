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
#include <iostream>
#include <chrono>
#include <mpi.h>
#include <omp.h>

// Global variables
int* image;
int rank, proc_num, thread_num;
int iters, width, height, calc_width, calc_height;
int curr_x_start, curr_y_start, proc_y_start, proc_y_end;
double left, right, lower, upper, x_step, y_step;

// Thread struct
struct thread_data {
    int thread_id;
    int iter;
    double runtime;
};

// Set calculation parameters
void set_calc_wh() {
    calc_width = width;
    calc_height = 1000;
}

// Thread function
void* thread_func(void* t_data) {
    auto start = std::chrono::high_resolution_clock::now();
    int iter = 0;

    while(true) {
        iter++;
        // Declare variables
        int x_start, x_end, y_start, y_end;

        // Entry section
        pthread_mutex_t mutex;
        pthread_mutex_init(&mutex, NULL);
        pthread_mutex_lock(&mutex);

        // Critical section
        if(curr_y_start >= proc_y_end) {
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
        if(y_end > proc_y_end) {
            y_end = proc_y_end;
        }

        curr_x_start = x_end;
        if(curr_x_start >= width) {
            curr_x_start = 0;
            curr_y_start += calc_height;
        }

        // Exit section
        pthread_mutex_unlock(&mutex);
        pthread_mutex_destroy(&mutex);

        // Calculate
        std::cout << "Rank: " << rank << ", Thread " << ((thread_data*)t_data)->thread_id << " calculating from (" << x_start << ", " << y_start << ") to (" << x_end << ", " << y_end << ")" << std::endl;
        // calc_mandelbrot_set(x_start, x_end, y_start, y_end);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double thread_time = std::chrono::duration<double, std::milli>(end - start).count() / 1000;
    ((thread_data*)t_data)->iter = iter;
    ((thread_data*)t_data)->runtime = thread_time;

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
    // MPI init
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_num);

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
    std::cout << "Image size: " << width << "x" << height << "=" << width * height << std::endl;

    x_step = (right - left) / width;
    y_step = (upper - lower) / height;

    proc_y_start = rank * height / proc_num;
    proc_y_end = (rank + 1) * height / proc_num;
    std::cout << "rank: " << rank << ", proc_y_start: " << proc_y_start << ", proc_y_end: " << proc_y_end << ", size: " << proc_y_end - proc_y_start << std::endl;

    // Allocate memory for image
    image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    // Set calc width and height
    set_calc_wh();
    curr_x_start = 0;
    curr_y_start = proc_y_start;

    // Create threads
    thread_num = CPU_COUNT(&cpu_set);
    pthread_t threads[thread_num];
    int thread_id[thread_num];
    thread_data t_data[thread_num];
    for(int i = 0; i < thread_num; ++i) {
        thread_id[i] = i;
        t_data[i].thread_id = i;
        int rc = pthread_create(&threads[i], NULL, thread_func, (void*)(t_data + i));
        assert(rc == 0);
    }

    // Mandelbrot set
    // calc_mandelbrot_set(0, width, 0, height);
    for (int j = 0; j < height; ++j) {
        double y0 = j * y_step + lower;
        for (int i = 0; i < width; ++i) {
            double x0 = i * ((right - left) / width) + left;

            int repeats = 0;
            double x = 0;
            double y = 0;
            double lsqr = 0;
            while (repeats < iters && lsqr < 4) {
                double temp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = temp;
                lsqr = x * x + y * y;
                ++repeats;
            }
            image[j * width + i] = repeats;
        }
    }

    // Draw and cleanup
    write_png(filename, iters, width, height, image);
    free(image);
}
