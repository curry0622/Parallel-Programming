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
#include <chrono>
#include <emmintrin.h>
#include <smmintrin.h>
#include <pmmintrin.h>

// Global variables
int* image;
int thread_num;
int iters;
int width, height;
int curr_x_start = 0, curr_y_start = 0;
int calc_width = 128, calc_height = 128;
double left, right, lower, upper;
double x_step, y_step;

// Packed doubles
union pd {
    __m128d dd;
    double d[2];
};

// Thread struct
struct thread_data {
    int thread_id;
    int iter;
    double runtime;
};

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

// Calculate the length_squared using SSE and packed doubles
void calc_lsqr_sse_pd(pd* x, pd* y, pd* x0, pd* y0, pd* lsqr) {
    pd temp;
    temp.dd = _mm_add_pd(_mm_sub_pd(_mm_mul_pd(x->dd, x->dd), _mm_mul_pd(y->dd, y->dd)), x0->dd);
    y->dd = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(x->dd, y->dd), _mm_set1_pd(2)), y0->dd);
    x->dd = temp.dd;
    lsqr->dd = _mm_add_pd(_mm_mul_pd(x->dd, x->dd), _mm_mul_pd(y->dd, y->dd));
}

// Calculate the mandelbrot set
void calc_mandelbrot_set(int x_start, int x_end, int y_start, int y_end) {
    for(int j = y_start; j < y_end; ++j) {
        double y0 = j * y_step + lower;
        for(int i = x_start; i < x_end; ++i) {
            double x0 = i * x_step + left;
            double x = 0;
            double y = 0;
            double length_squared = 0;
            int repeats = 0;

            while(repeats < iters && length_squared < 4) {
                calc_lsqr(&x, &y, &x0, &y0, &length_squared);
                ++repeats;
            }

            image[j * width + i] = repeats;
        }
    }
}

// Calculate the mandelbrot set using SSE
void calc_mandelbrot_set_sse(int x_start, int x_end, int y_start, int y_end) {
    __m128d two = _mm_set_pd1(2);
    __m128d four = _mm_set_pd1(4);
    for(int j = y_start; j < y_end; ++j) {
        double y0 = j * y_step + lower;
        __m128d y00 = _mm_load1_pd(&y0); // y00 = [64(y0), 64(y0)]
        int x_end1 = (x_end >> 1) << 1;

        for(int i = x_start; i < x_end1; i += 2) {
            double x0[2] = {i * x_step + left, (i + 1) * x_step + left};
            __m128d x00 = _mm_load_pd(x0); // x00 = [64(i + 1), 64(i)]
            __m128d x = _mm_setzero_pd(); // x = [0..0]
            __m128d y = _mm_setzero_pd(); // y = [0..0]
            __m128d lsqr = _mm_setzero_pd(); // lsqr = [0..0]
            int repeats[2] = {0, 0};
            bool finish[2] = {false, false};

            while(!finish[0] || !finish[1]) {
                if(!finish[0]) {
                    if(repeats[0] < iters && _mm_comilt_sd(lsqr, four)) {
                        repeats[0]++;
                    } else {
                        finish[0] = true;
                    }
                }
                if(!finish[1]) {
                    if(repeats[1] < iters && _mm_comilt_sd(_mm_unpackhi_pd(lsqr, lsqr), four)) {
                        repeats[1]++;
                    } else {
                        finish[1] = true;
                    }
                }
                calc_lsqr_sse(&x, &y, &x00, &y00, &lsqr);
            }
            image[j * width + i] = repeats[0];
            image[j * width + i + 1] = repeats[1];
        }

        for(int i = x_end1; i < x_end; ++i) {
            double x0 = i * x_step + left;
            double x = 0;
            double y = 0;
            double lsqr = 0;
            int repeats = 0;

            while(repeats < iters && lsqr < 4) {
                calc_lsqr(&x, &y, &x0, &y0, &lsqr);
                ++repeats;
            }

            image[j * width + i] = repeats;
        }
    }
}

// Calculate the mandelbrot set using SSE with channel method
void calc_mandelbrot_set_sse_v2(int x_start, int x_end, int y_start, int y_end) {
    // Initialize
    int total_num = (x_end - x_start) * (y_end - y_start);
    int curr_idx = 0;
    int idx[2] = {0, 0};
    int repeats[2] = {0, 0};
    bool reset[2] = {true, true};
    bool ge[2] = {false, false}; // greater or equal
    bool finished[2] = {false, false};

    __m128d zero = _mm_setzero_pd(), two = _mm_set1_pd(2), four = _mm_set1_pd(4);
    __m128d x = zero;
    __m128d y = zero;
    __m128d x0 = zero;
    __m128d y0 = _mm_set_pd1(y_start * y_step + lower);
    __m128d lsqr = zero;

    while(!finished[0] || !finished[1]) {
        // Reset
        if(reset[0] && !finished[0]) { // right channel (lsb)
            reset[0] = false;
            repeats[0] = 0;
            ge[0] = false;
            idx[0] = curr_idx++;
            x = _mm_move_sd(x, zero);
            y = _mm_move_sd(y, zero);
            x0 = _mm_move_sd(x0, _mm_set_pd1((x_start + idx[0]) * x_step + left));
            lsqr = _mm_move_sd(lsqr, zero);

            if(curr_idx > total_num) {
                finished[0] = true;
            }
        }
        if(reset[1] && !finished[1]) { // left channel (msb)
            reset[1] = false;
            repeats[1] = 0;
            ge[1] = false;
            idx[1] = curr_idx++;
            x = _mm_move_sd(zero, x);
            y = _mm_move_sd(zero, y);
            x0 = _mm_move_sd(_mm_set_pd1((x_start + idx[1]) * x_step + left), x0);
            lsqr = _mm_move_sd(zero, lsqr);

            if(curr_idx > total_num) {
                finished[1] = true;
            }
        }

        // Calculation
        __m128d temp = _mm_add_pd(_mm_sub_pd(_mm_mul_pd(x, x), _mm_mul_pd(y, y)), x0);
        y = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(x, y), two), y0);
        x = temp;
        lsqr = _mm_add_pd(_mm_mul_pd(x, x), _mm_mul_pd(y, y));
        int cmpge = _mm_movemask_pd(_mm_cmpge_pd(lsqr, four));
        ge[0] = cmpge & 1, ge[1] = cmpge & 2;
        repeats[0]++, repeats[1]++;

        // Check and update
        if((ge[0] || repeats[0] >= iters) && !finished[0]) {
            image[y_start * width + (x_start + idx[0])] = repeats[0];
            reset[0] = true;
        }
        if((ge[1] || repeats[1] >= iters) && !finished[1]) {
            image[y_start * width + (x_start + idx[1])] = repeats[1];
            reset[1] = true;
        }
    }
}

// Calculate the mandelbrot set using SSE with channel method and packed doubles
void calc_mandelbrot_set_sse_v3(int x_start, int x_end, int y_start, int y_end) {
    // Initialize
    int curr_idx = 0;
    int idx[2] = {0, 0};
    int repeats[2] = {0, 0};
    bool reset[2] = {true, true};
    bool ge[2] = {false, false}; // greater or equal
    bool finished[2] = {false, false};

    pd x, y, x0, y0, lsqr;
    y0.d[0] = y_start * y_step + lower;
    y0.d[1] = y_start * y_step + lower;

    while(!finished[0] || !finished[1]) {
        // Reset
        if(reset[0] && !finished[0]) { // right channel (lsb)
            reset[0] = false;
            repeats[0] = 0;
            ge[0] = false;
            idx[0] = curr_idx++;
            x.d[0] = 0;
            y.d[0] = 0;
            x0.d[0] = (x_start + idx[0]) * x_step + left;
            lsqr.d[0] = 0;

            if(curr_idx > calc_width) {
                finished[0] = true;
            }
        }
        if(reset[1] && !finished[1]) { // left channel (msb)
            reset[1] = false;
            repeats[1] = 0;
            ge[1] = false;
            idx[1] = curr_idx++;
            x.d[1] = 0;
            y.d[1] = 0;
            x0.d[1] = (x_start + idx[1]) * x_step + left;
            lsqr.d[1] = 0;

            if(curr_idx > calc_width) {
                finished[1] = true;
            }
        }

        // Calc
        calc_lsqr_sse_pd(&x, &y, &x0, &y0, &lsqr);
        repeats[0]++, repeats[1]++;

        // Check and update
        if((lsqr.d[0] >= 4 || repeats[0] >= iters) && !finished[0]) {
            image[y_start * width + (x_start + idx[0])] = repeats[0];
            reset[0] = true;
        }
        if((lsqr.d[1] >= 4 || repeats[1] >= iters) && !finished[1]) {
            image[y_start * width + (x_start + idx[1])] = repeats[1];
            reset[1] = true;
        }
    }
}

// Calculate the mandelbrot set using SSE with channel method
void calc_mandelbrot_set_sse_v4(int x_start, int x_end, int y_start, int y_end) {
    // Initialize
    int total_num = calc_width;
    int curr_idx = 0;
    int idx[2] = {curr_idx++, curr_idx++};
    int repeats[2] = {0, 0};
    bool ge[2] = {false, false}; // greater or equal
    bool finished[2] = {false, false};

    __m128d zero = _mm_setzero_pd(), two = _mm_set1_pd(2), four = _mm_set1_pd(4);
    __m128d x = zero;
    __m128d y = zero;
    __m128d x0 = _mm_set_pd((x_start + idx[1]) * x_step + left, (x_start + idx[0]) * x_step + left);
    __m128d y0 = _mm_set_pd1(y_start * y_step + lower);
    __m128d lsqr = zero;

    while(!finished[0] || !finished[1]) {
        // Calculation
        __m128d temp = _mm_add_pd(_mm_sub_pd(_mm_mul_pd(x, x), _mm_mul_pd(y, y)), x0);
        y = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(x, y), two), y0);
        x = temp;
        lsqr = _mm_add_pd(_mm_mul_pd(x, x), _mm_mul_pd(y, y));
        int cmpge = _mm_movemask_pd(_mm_cmpge_pd(lsqr, four));
        ge[0] = cmpge & 1, ge[1] = cmpge & 2;
        repeats[0]++, repeats[1]++;

        // Check and update
        if((ge[0] || repeats[0] >= iters) && !finished[0]) {
            // Update image
            image[y_start * width + (x_start + idx[0])] = repeats[0];

            // Reset
            repeats[0] = 0;
            ge[0] = false;
            idx[0] = curr_idx++;
            x = _mm_move_sd(x, zero);
            y = _mm_move_sd(y, zero);
            x0 = _mm_move_sd(x0, _mm_set_pd1((x_start + idx[0]) * x_step + left));
            lsqr = _mm_move_sd(lsqr, zero);

            if(curr_idx > total_num) {
                finished[0] = true;
            }
        }
        if((ge[1] || repeats[1] >= iters) && !finished[1]) {
            // Update image
            image[y_start * width + (x_start + idx[1])] = repeats[1];

            // Reset
            repeats[1] = 0;
            ge[1] = false;
            idx[1] = curr_idx++;
            x = _mm_move_sd(zero, x);
            y = _mm_move_sd(zero, y);
            x0 = _mm_move_sd(_mm_set_pd1((x_start + idx[1]) * x_step + left), x0);
            lsqr = _mm_move_sd(zero, lsqr);

            if(curr_idx > total_num) {
                finished[1] = true;
            }
        }
    }
}

// Calculate the mandelbrot set using SSE with channel method
void calc_mandelbrot_set_sse_v5(int x_start, int x_end, int y_start, int y_end) {
    // Initialize
    int curr_idx = 0;
    int idx[2] = {curr_idx++, curr_idx++};
    int repeats[2] = {0, 0};
    bool ge[2] = {false, false}; // greater or equal
    bool finished[2] = {false, false};

    __m128d zero = _mm_setzero_pd(), two = _mm_set1_pd(2), four = _mm_set1_pd(4);
    __m128d x = zero;
    __m128d y = zero;
    __m128d x0 = _mm_set_pd((x_start + idx[1]) * x_step + left, (x_start + idx[0]) * x_step + left);
    __m128d y0 = _mm_set_pd1(y_start * y_step + lower);
    __m128d lsqr = zero;

    while(idx[0] < calc_width && idx[1] < calc_width) {
        bool ge[2] = {false, false}; // greater or equal
        while(true) {
            calc_lsqr_sse(&x, &y, &x0, &y0, &lsqr);
            repeats[0]++, repeats[1]++;
            int cmpge = _mm_movemask_pd(_mm_cmpge_pd(lsqr, four));
            ge[0] = cmpge & 1, ge[1] = cmpge & 2;
            // ge[0] = !_mm_comilt_sd(lsqr, four);
            // ge[1] = !_mm_comilt_sd(_mm_unpackhi_pd(lsqr, lsqr), four);

            if(ge[0] || repeats[0] >= iters || ge[1] || repeats[1] >= iters) {
                break;
            }
        }

        if(ge[0] || repeats[0] >= iters) {
            // Update image
            image[y_start * width + (x_start + idx[0])] = repeats[0];

            // Reset
            repeats[0] = 0;
            ge[0] = false;
            idx[0] = curr_idx++;
            x = _mm_move_sd(x, zero);
            y = _mm_move_sd(y, zero);
            x0 = _mm_move_sd(x0, _mm_set_pd1((x_start + idx[0]) * x_step + left));
            lsqr = _mm_move_sd(lsqr, zero);
        }
        if(ge[1] || repeats[1] >= iters) {
            // Update image
            image[y_start * width + (x_start + idx[1])] = repeats[1];

            // Reset
            repeats[1] = 0;
            ge[1] = false;
            idx[1] = curr_idx++;
            x = _mm_move_sd(zero, x);
            y = _mm_move_sd(zero, y);
            x0 = _mm_move_sd(_mm_set_pd1((x_start + idx[1]) * x_step + left), x0);
            lsqr = _mm_move_sd(zero, lsqr);
        }
    }

    if(idx[0] < calc_width) {
        calc_mandelbrot_set(x_start + idx[0], x_end, y_start, y_end);
    }
    if(idx[1] < calc_width) {
        calc_mandelbrot_set(x_start + idx[1], x_end, y_start, y_end);
    }
}

// Calculate the mandelbrot set using SSE with channel method and packed doubles
void calc_mandelbrot_set_sse_v6(int x_start, int x_end, int y_start, int y_end) {
    // Initialize
    int curr_idx = 0;
    int idx[2] = {curr_idx++, curr_idx++};
    int repeats[2] = {0, 0};
    bool reset[2] = {true, true};
    bool ge[2] = {false, false}; // greater or equal
    bool finished[2] = {false, false};
    double zero;
    double x0buf[2], y0buf;

    pd x, y, x0, y0, lsqr, temp;

    x0buf[0] = (x_start + idx[0]) * x_step + left;
    x0buf[1] = (x_start + idx[1]) * x_step + left;
    x0.dd = _mm_load_pd(x0buf);

    y0buf = y_start * y_step + lower;
    y0.dd = _mm_load_pd1(&y0buf);

    while(idx[0] < calc_width && idx[1] < calc_width) {
        while(true) {
            // Calc
            temp.dd = _mm_add_pd(_mm_sub_pd(_mm_mul_pd(x.dd, x.dd), _mm_mul_pd(y.dd, y.dd)), x0.dd);
            y.dd = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(x.dd, y.dd), _mm_set1_pd(2)), y0.dd);
            x.dd = temp.dd;
            lsqr.dd = _mm_add_pd(_mm_mul_pd(x.dd, x.dd), _mm_mul_pd(y.dd, y.dd));
            repeats[0]++, repeats[1]++;

            if(lsqr.d[0] > 4 || lsqr.d[1] > 4 || repeats[0] >= iters || repeats[1] >= iters) {
                break;
            }
        }

        // Check and update
        if(lsqr.d[0] > 4 || repeats[0] >= iters) {
            image[y_start * width + (x_start + idx[0])] = repeats[0];

            repeats[0] = 0;
            idx[0] = curr_idx++;
            x.dd = _mm_loadl_pd(x.dd, &zero);
            y.dd = _mm_loadl_pd(y.dd, &zero);
            x0buf[0] = (x_start + idx[0]) * x_step + left;
            lsqr.dd = _mm_loadl_pd(lsqr.dd, &zero);
        }
        if(lsqr.d[1] >= 4 || repeats[1] >= iters) {
            image[y_start * width + (x_start + idx[1])] = repeats[1];

            repeats[1] = 0;
            idx[1] = curr_idx++;
            x.dd = _mm_loadh_pd(x.dd, &zero);
            y.dd = _mm_loadh_pd(y.dd, &zero);
            x0buf[1] = (x_start + idx[1]) * x_step + left;
            lsqr.dd = _mm_loadh_pd(lsqr.dd, &zero);
        }
        x0.dd = _mm_load_pd(x0buf);
    }

    if(idx[0] < calc_width) {
        while(repeats[0] < iters && lsqr.d[0] < 4) {
            double tmp = x.d[0] * x.d[0] - y.d[0] * y.d[0] + x0.d[0];
            y.d[0] = 2 * x.d[0] * y.d[0] + y0.d[0];
            x.d[0] = tmp;
            lsqr.d[0] = x.d[0] * x.d[0] + y.d[0] * y.d[0];
            repeats[0]++;
        }
        image[y_start * width + (x_start + idx[0])] = repeats[0];
    }
    if(idx[1] < calc_width) {
        while(repeats[1] < iters && lsqr.d[1] < 4) {
            double tmp = x.d[1] * x.d[1] - y.d[1] * y.d[1] + x0.d[1];
            y.d[1] = 2 * x.d[1] * y.d[1] + y0.d[0];
            x.d[1] = tmp;
            lsqr.d[1] = x.d[1] * x.d[1] + y.d[1] * y.d[1];
            repeats[1]++;
        }
        image[y_start * width + (x_start + idx[1])] = repeats[1];
    }
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
            curr_y_start += calc_height;
        }

        // Exit section
        pthread_mutex_unlock(&mutex);
        pthread_mutex_destroy(&mutex);

        // Calculate
        // calc_mandelbrot_set(x_start, x_end, y_start, y_end);
        // calc_mandelbrot_set_sse(x_start, x_end, y_start, y_end);
        // calc_mandelbrot_set_sse_v2(x_start, x_end, y_start, y_end);
        // calc_mandelbrot_set_sse_v3(x_start, x_end, y_start, y_end);
        // calc_mandelbrot_set_sse_v4(x_start, x_end, y_start, y_end);
        calc_mandelbrot_set_sse_v5(x_start, x_end, y_start, y_end);
        // calc_mandelbrot_set_sse_v6(x_start, x_end, y_start, y_end);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double thread_time = std::chrono::duration<double, std::milli>(end - start).count() / 1000;
    ((thread_data*)t_data)->iter = iter;
    ((thread_data*)t_data)->runtime = thread_time;

    // Exit
    pthread_exit(NULL);
}

// Write the image to a png file
void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    auto start = std::chrono::high_resolution_clock::now();
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
    for(int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for(int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if(p != iters) {
                if(p & 16) {
                    color[0] = 240;
                    // color[1] = color[2] = p % 16 * 16;
                    color[1] = color[2] = (p & 15) << 4;
                } else {
                    // color[0] = p % 16 * 16;
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
    auto end = std::chrono::high_resolution_clock::now();
    double img_time = std::chrono::duration<double, std::milli>(end - start).count() / 1000;
    std::cout << "Image write time: " << img_time << " seconds" << std::endl;
}

// Set calculation parameters
void set_calc_wh() {
    calc_width = width;
    calc_height = 1;
}

// Write image info
void write_img_info() {
    FILE* fp = fopen("v1.txt", "w");
    assert(fp);
    for(int i = 0; i < width * height; i++) {
        fprintf(fp, "%d: %d\n", i, image[i]);
    }
    fclose(fp);
}

int main(int argc, char** argv) {
    auto program_start = std::chrono::high_resolution_clock::now();
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
    std::cout << "Image size: " << width << " x " << height << " = " << width * height << std::endl;

    x_step = (right - left) / width;
    y_step = (upper - lower) / height;

    // Allocate memory for image
    image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    // Set calc width and height
    set_calc_wh();

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

    // Join threads
    for(int i = 0; i < thread_num; ++i) {
        pthread_join(threads[i], NULL);
        std::cout << "Thread " << i << " runtime: " << t_data[i].runtime << " seconds, iteration: " << t_data[i].iter << std::endl;
    }

    // Write image info
    // write_img_info();

    // Draw and cleanup
    write_png(filename, iters, width, height, image);
    free(image);
    auto program_end = std::chrono::high_resolution_clock::now();
    double elapsed_time = std::chrono::duration<double, std::milli>(program_end - program_start).count() / 1000;

    // Print time
    std::cout << "Elapsed time: " << elapsed_time << " s" << std::endl;
}
