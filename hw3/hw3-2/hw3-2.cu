#include <stdio.h>
#include <stdlib.h>

const int INF = ((1 << 30) - 1);
const int V = 50010;

int n, m; // n: # of vertices, m: # of edges
int* h_dist;

__device__ __host__ int convert_index(int i, int j, int row_size) {
    return i * row_size + j;
}

/* Read file input */
void input(char* infile) {
    // Read n and m
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    // Allocate memory for h_dist
    cudaMallocHost((void**)&h_dist, sizeof(int) * n * n);

    // Initialize h_dist
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int idx = convert_index(i, j, n);
            if (i == j) {
                h_dist[idx] = 0;
            } else {
                h_dist[idx] = INF;
            }
        }
    }

    // Read edges
    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        int idx = convert_index(pair[0], pair[1], n);
        h_dist[idx] = pair[2];
    }
    fclose(file);
}

/* Write file output */
void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int idx = convert_index(i, j, n);
            if(h_dist[idx] >= INF)
                h_dist[idx] = INF;
        }
        fwrite(h_dist + i * n, sizeof(int), n, outfile);
    }
    fclose(outfile);
}

int ceil(int a, int b) {
    return (a + b - 1) / b;
}

void cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height) {
    int block_end_x = block_start_x + block_width;
    int block_end_y = block_start_y + block_height;
    int end_k = (Round + 1) * B > n ? n : (Round + 1) * B;

    for (int b_i = block_start_x; b_i < block_end_x; ++b_i) {
        for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {
            // To calculate B*B elements in the block (b_i, b_j)
            // For each block, it need to compute B times
            int block_internal_start_x = b_i * B;
            int block_internal_end_x = (b_i + 1) * B;
            int block_internal_start_y = b_j * B;
            int block_internal_end_y = (b_j + 1) * B;
            if (block_internal_end_x > n) block_internal_end_x = n;
            if (block_internal_end_y > n) block_internal_end_y = n;

            for (int k = Round * B; k < end_k; ++k) {
                // To calculate original index of elements in the block (b_i, b_j)
                // For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2
                #pragma omp parallel for schedule(dynamic, 1)
                for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
                    for (int j = block_internal_start_y; j < block_internal_end_y; ++j) {
                        int i_k = convert_index(i, k, n);
                        int k_j = convert_index(k, j, n);
                        int i_j = convert_index(i, j, n);
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
extern __shared__ int s_dist[];
__global__ void phase1(int* d_dist, int B, int r, int n) {
    // Get index of thread
    int i = threadIdx.x;
    int j = threadIdx.y;
    int s_idx = convert_index(i, j, B);
    int h_idx = convert_index(i + r * B, j + r * B, n);

    // Copy data from global memory to shared memory
    s_dist[s_idx] = d_dist[h_idx];

    // Compute
    for(int k = 0; k < B; ++k) {
        __syncthreads();
        int i_k_dist = s_dist[convert_index(i, k, B)];
        int k_j_dist = s_dist[convert_index(k, j, B)];
        if (i_k_dist + k_j_dist < s_dist[s_idx]) {
            s_dist[s_idx] = i_k_dist + k_j_dist;
        }
    }

    // Copy data from shared memory to global memory
    d_dist[h_idx] = s_dist[s_idx];
}

void block_FW(int B, int* d_dist) {
    int round = ceil(n, B);
    for (int r = 0; r < round; ++r) {
        printf("Round %d\n", r);
        /* Phase 1*/
        // cal(B, r, r, r, 1, 1);
        // cudaMemcpy(d_dist, h_dist, sizeof(int) * n * n, cudaMemcpyHostToDevice);
        // dim3 thds_per_blk(B, B);
        // phase1<<<1, thds_per_blk, B * B * sizeof(int)>>>(d_dist, B, r, n);
        // cudaMemcpy(h_dist, d_dist, n * n * sizeof(int), cudaMemcpyDeviceToHost);

        // FILE* file = fopen("output0.txt", "a");
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                printf("%d, %d -> %d\n", i, j, h_dist[i * n + j]);
                // fprintf(file, "%d, %d -> %d\n", i, j, h_dist[i * n + j]);
            }
        }
        // fclose(file);

        /* Phase 2*/
        cal(B, r, r, 0, 1, r);
        cal(B, r, r, r + 1, 1, round - r - 1);
        cal(B, r, 0, r, r, 1);
        cal(B, r, r + 1, r, round - r - 1, 1);

        /* Phase 3*/
        cal(B, r, 0, 0, r, r);
        cal(B, r, 0, r + 1, r, round - r - 1);
        cal(B, r, r + 1, 0, round - r - 1, r);
        cal(B, r, r + 1, r + 1, round - r - 1, round - r - 1);
    }
}

int main(int argc, char* argv[]) {
    // Read input
    printf("Reading input...\n");
    input(argv[1]);
    printf("Read input done.\n");

    // Allocate memory for d_dist
    printf("Allocating memory...\n");
    int* d_dist;
    cudaMalloc((void**)&d_dist, sizeof(int) * n * n);
    printf("Allocate memory done.\n");

    // Copy data from host to device
    printf("Copying data...\n");
    cudaMemcpy(d_dist, h_dist, sizeof(int) * n * n, cudaMemcpyHostToDevice);
    printf("Copy data done.\n");

    // Block FW
    printf("Block FW...\n");
    int B = 32;
    block_FW(B, d_dist);
    printf("Block FW done.\n");

    // Write output
    printf("Writing output...\n");
    output(argv[2]);
    printf("Write output done.\n");
    return 0;
}
