#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define GPU_NUM 2
#define BLK_FAC 32

const int INF = ((1 << 30) - 1);
const int V = 50010;

int vtx_num, edge_num, mtx_size;
int* h_dist;

__constant__ int d_vtx_num, d_mtx_size, d_blk_fac;

__device__ __host__ int convert_index(int i, int j, int row_size) {
    return i * row_size + j;
}

/* Get ceil(a / b) */
int ceil(int a, int b) {
    return (a + b - 1) / b;
}

/* Read file input */
void input(char* infile) {
    // Read vertex num and edge num
    FILE* file = fopen(infile, "rb");
    fread(&vtx_num, sizeof(int), 1, file);
    fread(&edge_num, sizeof(int), 1, file);

    // Calculate matrix size
    mtx_size = ceil(vtx_num, BLK_FAC) * BLK_FAC;

    // Allocate memory for h_dist
    cudaMallocHost((void**)&h_dist, sizeof(int) * mtx_size * mtx_size);

    // Initialize h_dist
    for (int i = 0; i < mtx_size; ++i) {
        for (int j = 0; j < mtx_size; ++j) {
            int idx = convert_index(i, j, mtx_size);
            if(i == j && i < vtx_num && j < vtx_num)
                h_dist[idx] = 0;
            else
                h_dist[idx] = INF;
        }
    }

    // Read edges
    int pair[3];
    for (int i = 0; i < edge_num; ++i) {
        fread(pair, sizeof(int), 3, file);
        int idx = convert_index(pair[0], pair[1], mtx_size);
        h_dist[idx] = pair[2];
    }
    fclose(file);
}

/* Write file output */
void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < vtx_num; ++i) {
        for (int j = 0; j < vtx_num; ++j) {
            int idx = convert_index(i, j, mtx_size);
            if(h_dist[idx] >= INF)
                h_dist[idx] = INF;
        }
        fwrite(h_dist + i * mtx_size, sizeof(int), vtx_num, outfile);
    }
    fclose(outfile);
}

/* Phase 1's kernel */
__global__ void phase1(int* d_dist, int r) {
    // Get index of thread
    int j = threadIdx.x; // col idx
    int i = threadIdx.y; // row idx
    int s_idx = convert_index(i, j, d_blk_fac);
    int h_idx = convert_index(i + r * d_blk_fac, j + r * d_blk_fac, d_mtx_size);

    // Copy data from global memory to shared memory
    extern __shared__ int s_mem[];
    s_mem[s_idx] = d_dist[h_idx];

    // Compute (unroll BLK_FAC times)
    #pragma unroll 32
    for(int k = 0; k < d_blk_fac; ++k) {
        __syncthreads();
        int i_k_dist = s_mem[convert_index(i, k, d_blk_fac)];
        int k_j_dist = s_mem[convert_index(k, j, d_blk_fac)];
        if (i_k_dist + k_j_dist < s_mem[s_idx]) {
            s_mem[s_idx] = i_k_dist + k_j_dist;
        }
    }

    // Copy data from shared memory to global memory
    d_dist[h_idx] = s_mem[s_idx];
}

/* Phase 2's kernel */
__global__ void phase2(int* d_dist, int r) {
    // Get index of thread
    int j = threadIdx.x; // col idx
    int i = threadIdx.y; // row idx
    int real_i, real_j;
    int s_idx = convert_index(i, j, d_blk_fac), h_idx;
    int blk_size = d_blk_fac * d_blk_fac;
    int i_k_offset = 0, k_j_offset = 0;

    // Copy data from global memory to shared memory
    if(blockIdx.x == 0) {
        // Pivot row
        i_k_offset = blk_size;
        if(blockIdx.y < r) {
            // Left blks of pivot blk
            real_i = i + r * d_blk_fac;
            real_j = j + blockIdx.y * d_blk_fac;
        } else {
            // Right blks of pivot blk
            real_i = i + r * d_blk_fac;
            real_j = j + (blockIdx.y + 1) * d_blk_fac;
        }
    } else {
        // Pivot col
        k_j_offset = blk_size;
        if(blockIdx.y < r) {
            // Up blks of pivot blk
            real_i = i + blockIdx.y * d_blk_fac;
            real_j = j + r * d_blk_fac;
        } else {
            // Down blks of pivot blk
            real_i = i + (blockIdx.y + 1) * d_blk_fac;
            real_j = j + r * d_blk_fac;
        }
    }
    h_idx = convert_index(real_i, real_j, d_mtx_size);

    extern __shared__ int s_mem[];
    s_mem[s_idx] = d_dist[h_idx]; // curr blk
    s_mem[blk_size + s_idx] = d_dist[convert_index(i + r * d_blk_fac, j + r * d_blk_fac, d_mtx_size)]; // pivot blk

    // Compute (unroll BLK_FAC times)
    #pragma unroll 32
    for(int k = 0; k < d_blk_fac; ++k) {
        __syncthreads();
        int i_k_dist = s_mem[i_k_offset + convert_index(i, k, d_blk_fac)];
        int k_j_dist = s_mem[k_j_offset + convert_index(k, j, d_blk_fac)];
        if (i_k_dist + k_j_dist < s_mem[s_idx]) {
            s_mem[s_idx] = i_k_dist + k_j_dist;
        }
    }

    // Copy data from shared memory to global memory
    d_dist[h_idx] = s_mem[s_idx];
}

/* Phase 3's kernel */
__global__ void phase3(int* d_dist, int r, int d_id) {
    // Get index of thread
    int j = threadIdx.x; // col idx
    int i = threadIdx.y; // row idx
    int real_i, real_j;
    int blk_i, blk_j;
    int s_idx = convert_index(i, j, d_blk_fac), h_idx;
    int blk_size = d_blk_fac * d_blk_fac;

    // Copy data from global memory to shared memory
    if(blockIdx.x < r) {
        blk_j = blockIdx.x;
    } else {
        blk_j = blockIdx.x + 1;
    }
    real_j = j + blk_j * d_blk_fac;

    if(blockIdx.y < r) {
        blk_i = blockIdx.y;
    } else {
        blk_i = blockIdx.y + 1;
    }
    real_i = i + blk_i * d_blk_fac;

    if((d_id == 0 && blk_i >= r) || (d_id == 1 && blk_i < r)) {
        return;
    }

    h_idx = convert_index(real_i, real_j, d_mtx_size);

    extern __shared__ int s_mem[];
    s_mem[s_idx] = d_dist[h_idx]; // curr blk
    s_mem[blk_size + s_idx] = d_dist[convert_index(i + blk_i * d_blk_fac, j + r * d_blk_fac, d_mtx_size)]; // pivot row blk
    s_mem[2 * blk_size + s_idx] = d_dist[convert_index(i + r * d_blk_fac, j + blk_j * d_blk_fac, d_mtx_size)]; // pivot col blk

    // Compute (unroll BLK_FAC times)
    #pragma unroll 32
    for(int k = 0; k < d_blk_fac; ++k) {
        __syncthreads();
        int i_k_dist = s_mem[blk_size + convert_index(i, k, d_blk_fac)]; // pivot row blk
        int k_j_dist = s_mem[2 * blk_size + convert_index(k, j, d_blk_fac)]; // pivot col blk
        if (i_k_dist + k_j_dist < s_mem[s_idx]) {
            s_mem[s_idx] = i_k_dist + k_j_dist;
        }
    }

    // Copy data from shared memory to global memory
    d_dist[h_idx] = s_mem[s_idx];
}

int main(int argc, char* argv[]) {
    // Read input
    input(argv[1]);

    // Declare variables
    int* d_dist[GPU_NUM];

    // Block FW using 2 GPUs
    #pragma omp parallel num_threads(GPU_NUM)
    {
        // Set device
        int d_id = omp_get_thread_num(); // device id
        cudaSetDevice(d_id);

        // Allocate memory for constants
        int blk_fac = BLK_FAC;
        cudaMemcpyToSymbol(d_vtx_num, &vtx_num, sizeof(int));
        cudaMemcpyToSymbol(d_mtx_size, &mtx_size, sizeof(int));
        cudaMemcpyToSymbol(d_blk_fac, &blk_fac, sizeof(int));

        // Allocate memory for d_dist
        cudaMalloc((void**)&d_dist[d_id], sizeof(int) * mtx_size * mtx_size);

        // Copy data from host to device
        cudaMemcpy(d_dist[d_id], h_dist, sizeof(int) * mtx_size * mtx_size, cudaMemcpyHostToDevice);

        // Block FW
        int round = ceil(vtx_num, BLK_FAC);
        int s_mem_size = BLK_FAC * BLK_FAC * sizeof(int);
        dim3 thds_per_blk(BLK_FAC, BLK_FAC);
        dim3 p2_blks_per_grid(2, round - 1); // 2: 1 for row, 1 for col; round - 1: # of (blks in row(or col) - pivot blk)
        dim3 p3_blks_per_grid(round - 1, round - 1);

        for (int r = 0; r < round; ++r) {
            // GPU[1] send row[r] to GPU[0]
            if (d_id == 1) {
                cudaMemcpy(
                    d_dist[0] + convert_index(r * BLK_FAC, 0, mtx_size),
                    d_dist[1] + convert_index(r * BLK_FAC, 0, mtx_size),
                    sizeof(int) * BLK_FAC * mtx_size,
                    cudaMemcpyDeviceToDevice
                );
            }

            // Synchronize
            cudaDeviceSynchronize();
            #pragma omp barrier

            // Phases
            phase1<<<1, thds_per_blk, s_mem_size>>>(d_dist[d_id], r);
            phase2<<<p2_blks_per_grid, thds_per_blk, 2 * s_mem_size>>>(d_dist[d_id], r);
            phase3<<<p3_blks_per_grid, thds_per_blk, 3 * s_mem_size>>>(d_dist[d_id], r, d_id);
        }

        // Copy data from device to host
        if(d_id == 0) {
            cudaMemcpy(h_dist, d_dist[0], sizeof(int) * mtx_size * mtx_size, cudaMemcpyDeviceToHost);
        }
    }

    // Write output
    output(argv[2]);

    return 0;
}
