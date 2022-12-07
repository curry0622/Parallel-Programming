#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>

/* Constants */
const int INF = ((1 << 30) - 1);
const int V = 50010;

/* Global variables */
int n, m; // n: # of vertices, m: # of edges
int* dist;

//======================
#define DEV_NO 0
cudaDeviceProp prop;

/* Convert index */
int convert_index(int i, int j, int n) {
    return i * n + j;
}

/* Read input */
void input(char* infile) {
    // Read # of vertices and # of edges
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    // Allocate pinned memory for dist
    cudaError_t stat = cudaMallocHost((void**)&dist, sizeof(int) * n * n);
    if(stat != cudaSuccess) {
        std::cout << "Error: " << cudaGetErrorString(stat) << std::endl;
        exit(-1);
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << i << " " << j << std::endl;
            std::cout << convert_index(i, j, n) << std::endl;
            if (i == j) {
                dist[convert_index(i, j, n)] = 0;
            } else {
                dist[convert_index(i, j, n)] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        dist[convert_index(pair[0], pair[1], n)] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int idx = convert_index(i, j, n);
            if (dist[idx] >= INF) {
                dist[idx] = INF;
            }
        }
        fwrite(dist + i * n, sizeof(int), n, outfile);
    }
    fclose(outfile);
}

int ceil(int a, int b) {
    return (a + b - 1) / b;
}

int main(int argc, char* argv[]) {
    input(argv[1]);
    int B = 512;

    // cudaGetDeviceProperties(&prop, DEV_NO);
    // printf("maxThreadsPerBlock = %d, sharedMemPerBlock = %d\n", prop.maxThreadsPerBlock, prop.sharedMemPerBlock);

    // block_FW(B);
    output(argv[2]);
    return 0;
}