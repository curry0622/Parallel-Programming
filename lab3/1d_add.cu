#include <iostream>
#include <cstdlib>
#include <cassert>
#include <zlib.h>

__global__ void add(int *a, int *b, int *c) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main() {
    int a[2] = {130, 120}, b[2] = {210, 290}, c[2] = {0, 0}; // host copies of a, b, c
    int *d_a, *d_b, *d_c; // device copies of a, b, c

    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, 2 * sizeof(int));
    cudaMalloc((void **)&d_b, 2 * sizeof(int));
    cudaMalloc((void **)&d_c, 2 * sizeof(int));

    // Copy inputs to device
    cudaMemcpy(d_a, a, 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, 2 * sizeof(int), cudaMemcpyHostToDevice);

    // Launch add() kernel on GPU
    add<<<1,1024>>>(d_a, d_b, d_c);

    // Copy result back to host
    cudaMemcpy(c, d_c, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << c[0] << " " << c[1] << std::endl;

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}