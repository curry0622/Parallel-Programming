#include <stdio.h>
#include <stdlib.h>

#define BLK_FAC 2

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
    printf("vtx_num: %d\n", vtx_num);
    printf("blk_fac: %d\n", BLK_FAC);
    printf("mtx_size: %d\n", mtx_size);

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

void cal(int Round, int block_start_x, int block_start_y, int block_width, int block_height) {
    int B = BLK_FAC;
    int block_end_x = block_start_x + block_width;
    int block_end_y = block_start_y + block_height;
    int end_k = (Round + 1) * B > vtx_num ? vtx_num : (Round + 1) * B;

    for (int b_i = block_start_x; b_i < block_end_x; ++b_i) {
        for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {
            // To calculate B*B elements in the block (b_i, b_j)
            // For each block, it need to compute B times
            int block_internal_start_x = b_i * B;
            int block_internal_end_x = (b_i + 1) * B;
            int block_internal_start_y = b_j * B;
            int block_internal_end_y = (b_j + 1) * B;
            if (block_internal_end_x > vtx_num) block_internal_end_x = vtx_num;
            if (block_internal_end_y > vtx_num) block_internal_end_y = vtx_num;

            for (int k = Round * B; k < end_k; ++k) {
                // To calculate original index of elements in the block (b_i, b_j)
                // For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2
                #pragma omp parallel for schedule(dynamic, 1)
                for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
                    for (int j = block_internal_start_y; j < block_internal_end_y; ++j) {
                        int i_k = convert_index(i, k, mtx_size);
                        int k_j = convert_index(k, j, mtx_size);
                        int i_j = convert_index(i, j, mtx_size);
                        if (h_dist[i_k] + h_dist[k_j] < h_dist[i_j]) {
                            h_dist[i_j] = h_dist[i_k] + h_dist[k_j];
                        }
                    }
                }
            }
        }
    }
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

    // Compute
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

    // Copy data from global memory to shared memory
    if(blockIdx.x == 0) {
        // Pivot row
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

    // Compute
    for(int k = 0; k < d_blk_fac; ++k) {
        __syncthreads();
        int i_k_dist = s_mem[blk_size + convert_index(i, k, d_blk_fac)]; // element in pivot blk
        int k_j_dist = s_mem[convert_index(k, j, d_blk_fac)]; // element in curr blk
        if (i_k_dist + k_j_dist < s_mem[s_idx]) {
            s_mem[s_idx] = i_k_dist + k_j_dist;
        }
    }

    // Copy data from shared memory to global memory
    d_dist[h_idx] = s_mem[s_idx];
}

/* Phase 2 row kernel */
__global__ void phase2_row(int* d_dist, int r) {
    // Get index of thread
    int j = threadIdx.x; // col idx
    int i = threadIdx.y; // row idx
    int real_i, real_j;
    int s_idx = convert_index(i, j, d_blk_fac), h_idx;
    int blk_size = d_blk_fac * d_blk_fac;

    // Copy data from global memory to shared memory
    if(blockIdx.x < r) {
        // Left blks of pivot blk
        real_i = i + r * d_blk_fac;
        real_j = j + blockIdx.x * d_blk_fac;
    } else {
        // Right blks of pivot blk
        real_i = i + r * d_blk_fac;
        real_j = j + (blockIdx.x + 1) * d_blk_fac;
    }
    h_idx = convert_index(real_i, real_j, d_mtx_size);

    extern __shared__ int s_mem[];
    s_mem[s_idx] = d_dist[h_idx]; // curr blk
    s_mem[blk_size + s_idx] = d_dist[convert_index(i + r * d_blk_fac, j + r * d_blk_fac, d_mtx_size)]; // pivot blk

    // Compute
    for(int k = 0; k < d_blk_fac; ++k) {
        __syncthreads();
        int i_k_dist = s_mem[blk_size + convert_index(i, k, d_blk_fac)]; // element in pivot blk
        int k_j_dist = s_mem[convert_index(k, j, d_blk_fac)]; // element in curr blk
        if (i_k_dist + k_j_dist < s_mem[s_idx]) {
            s_mem[s_idx] = i_k_dist + k_j_dist;
        }
    }

    // Copy data from shared memory to global memory
    d_dist[h_idx] = s_mem[s_idx];
}

/* Phase 2 col kernel */
__global__ void phase2_col(int* d_dist, int r) {
    // Get index of thread
    int j = threadIdx.x; // col idx
    int i = threadIdx.y; // row idx
    int real_i, real_j;
    int s_idx = convert_index(i, j, d_blk_fac), h_idx;
    int blk_size = d_blk_fac * d_blk_fac;

    // Copy data from global memory to shared memory
    if(blockIdx.x < r) {
        // Up blks of pivot blk
        real_i = i + blockIdx.x * d_blk_fac;
        real_j = j + r * d_blk_fac;
    } else {
        // Down blks of pivot blk
        real_i = i + (blockIdx.x + 1) * d_blk_fac;
        real_j = j + r * d_blk_fac;
    }
    h_idx = convert_index(real_i, real_j, d_mtx_size);

    extern __shared__ int s_mem[];
    s_mem[s_idx] = d_dist[h_idx]; // curr blk
    s_mem[blk_size + s_idx] = d_dist[convert_index(i + r * d_blk_fac, j + r * d_blk_fac, d_mtx_size)]; // pivot blk

    // Compute
    for(int k = 0; k < d_blk_fac; ++k) {
        __syncthreads();
        int i_k_dist = s_mem[convert_index(i, k, d_blk_fac)]; // element in curr blk
        int k_j_dist = s_mem[blk_size + convert_index(k, j, d_blk_fac)]; // element in pivot blk
        if (i_k_dist + k_j_dist < s_mem[s_idx]) {
            s_mem[s_idx] = i_k_dist + k_j_dist;
        }
    }

    // Copy data from shared memory to global memory
    d_dist[h_idx] = s_mem[s_idx];
}

void block_FW(int* d_dist) {
    int round = ceil(vtx_num, BLK_FAC);
    int s_mem_size = BLK_FAC * BLK_FAC * sizeof(int);
    dim3 thds_per_blk(BLK_FAC, BLK_FAC);
    dim3 p2_blks_per_grid(2, round - 1); // 2: 1 for row, 1 for col; round - 1: # of (blks in row(or col) - pivot blk)

    for (int r = 0; r < round; ++r) {
        printf("Round %d\n", r);
        /* Phase 1*/
        // cal(r, r, r, 1, 1);
        cudaMemcpy(d_dist, h_dist, sizeof(int) * mtx_size * mtx_size, cudaMemcpyHostToDevice);
        phase1<<<1, thds_per_blk, s_mem_size>>>(d_dist, r);
        cudaMemcpy(h_dist, d_dist, mtx_size * mtx_size * sizeof(int), cudaMemcpyDeviceToHost);

        for(int i = 0; i < vtx_num; i++) {
            for(int j = 0; j < vtx_num; j++) {
                printf("%d, %d -> %d\n", i, j, h_dist[convert_index(i, j, mtx_size)]);
            }
        }

        /* Phase 2*/
        // cal(r, r, 0, 1, r);
        // cal(r, r, r + 1, 1, round - r - 1);
        // cal(r, 0, r, r, 1);
        // cal(r, r + 1, r, round - r - 1, 1);

        cudaMemcpy(d_dist, h_dist, sizeof(int) * mtx_size * mtx_size, cudaMemcpyHostToDevice);
        phase2_row<<<round - 1, thds_per_blk, 2 * s_mem_size>>>(d_dist, r);
        phase2_col<<<round - 1, thds_per_blk, 2 * s_mem_size>>>(d_dist, r);
        cudaMemcpy(h_dist, d_dist, mtx_size * mtx_size * sizeof(int), cudaMemcpyDeviceToHost);

        // // FILE* file = fopen("output0.txt", "a");
        for(int i = 0; i < vtx_num; i++) {
            for(int j = 0; j < vtx_num; j++) {
                printf("%d, %d -> %d\n", i, j, h_dist[convert_index(i, j, mtx_size)]);
                // fprintf(file, "%d, %d -> %d\n", i, j, h_dist[i * n + j]);
            }
        }
        // fclose(file);
        // break;

        /* Phase 3*/
        cal(r, 0, 0, r, r);
        cal(r, 0, r + 1, r, round - r - 1);
        cal(r, r + 1, 0, round - r - 1, r);
        cal(r, r + 1, r + 1, round - r - 1, round - r - 1);
    }
}

int main(int argc, char* argv[]) {
    // Read input
    printf("Reading input...\n");
    input(argv[1]);
    printf("Read input done.\n");

    // Allocate memory for constants
    printf("Allocating memory for constants...\n");
    int blk_fac = BLK_FAC;
    cudaMemcpyToSymbol(d_vtx_num, &vtx_num, sizeof(int));
    cudaMemcpyToSymbol(d_mtx_size, &mtx_size, sizeof(int));
    cudaMemcpyToSymbol(d_blk_fac, &blk_fac, sizeof(int));
    printf("Allocate memory for constants done.\n");

    // Allocate memory for d_dist
    printf("Allocating memory...\n");
    int* d_dist;
    cudaMalloc((void**)&d_dist, sizeof(int) * mtx_size * mtx_size);
    printf("Allocate memory done.\n");

    // Copy data from host to device
    printf("Copying data...\n");
    cudaMemcpy(d_dist, h_dist, sizeof(int) * mtx_size * mtx_size, cudaMemcpyHostToDevice);
    printf("Copy data done.\n");

    // Block FW
    printf("Block FW...\n");
    block_FW(d_dist);
    printf("Block FW done.\n");

    // Write output
    printf("Writing output...\n");
    output(argv[2]);
    printf("Write output done.\n");
    return 0;
}
