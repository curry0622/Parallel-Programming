#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <mpi.h>

#define BUF_SIZE 200000000

int cmp (const void * a, const void * b) {
   float fa = *(float *)a;
   float fb = *(float *)b;
   return (fa > fb) ? 1 : -1;
}

int getOffset(int rank, int arrSize, int procNum, bool overflow) {
    if(overflow)
        return BUF_SIZE * rank;
    int offset = 0;
    int left = arrSize % procNum;
    int base = arrSize / procNum;
    if (rank < left)
        offset = rank * (base + 1);
    else
        offset = left * (base + 1) + (rank - left) * base;
    return offset;
}

int main(int argc, char** argv) {
    // MPI init
    int rank, procNum;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);
    // printf("[Rank %d] MPI init done\n", rank);

    // Get args
    int arrSize = atoi(argv[1]);
    char* inFileName = argv[2];
    char* outFileName = argv[3];
    // printf("[Rank %d] Get args done\n", rank);

    // Set cut size
    int cutSize = arrSize / procNum;
    int left = arrSize % procNum;
    if (rank < left)
        cutSize++;

    // Try
    bool overflow = true;
    int base = 0;
    while(overflow) {
        int handleSize, offset;
        if(cutSize > BUF_SIZE) {
            handleSize = BUF_SIZE;
            cutSize -= BUF_SIZE;
            overflow = true;
        } else {
            handleSize = cutSize;
            overflow = false;
        }
        offset = getOffset(rank, arrSize - base, procNum, overflow);
        // printf("[Rank %d] offset: %d\n", rank, offset);
        
        // MPI read file
        // float data[handleSize];
        float* data = (float*)malloc(sizeof(float) * handleSize);
        // printf("[Rank %d] declare data done\n", rank);
        MPI_File fin;
        MPI_File_open(MPI_COMM_WORLD, inFileName, MPI_MODE_RDONLY, MPI_INFO_NULL, &fin);
        MPI_File_read_at_all(fin, base * sizeof(float) + offset * sizeof(float), data, handleSize, MPI_FLOAT, MPI_STATUS_IGNORE);
        MPI_File_close(&fin);
        // printf("[Rank %d] MPI read file done\n", rank);
        // for(int i = 0; i < procNum; i++) {
        //     if(i == rank) {
        //         printf("[Rank %d] data: ", rank);
        //         for(int j = 0; j < handleSize; j++)
        //             printf("%f ", data[j]);
        //         printf("\n");
        //     }
        // }

        // Initial sort
        qsort(data, handleSize, sizeof(float), cmp);
        MPI_Barrier(MPI_COMM_WORLD);
        // printf("[Rank %d] Initial sort done\n", rank);

        // Odd-Even-Sort
        bool isSortedAll = false;
        bool isOddStage = true;
        while(!isSortedAll) {
            // if(isOddStage)
            //     printf("[Rank %d] Odd stage start\n", rank);
            // else
            //     printf("[Rank %d] Even stage start\n", rank);
            bool isSorted = true;
            if(rank >= arrSize)
                goto BARRIER;
            if(isOddStage) {
                if(rank % 2 == 0) {
                    if(rank > 0) {
                        // printf("[Rank: %d] send handleSize to %d\n", rank, rank - 1);
                        MPI_Send(&handleSize, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
                        // printf("[Rank %d] send data to %d\n", rank, rank - 1);
                        MPI_Send(data , handleSize , MPI_FLOAT , rank - 1 , 0, MPI_COMM_WORLD);
                        // printf("[Rank %d] get handleSize from %d\n", rank, rank - 1);
                        MPI_Recv(data, handleSize, MPI_FLOAT, rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    } else {
                        goto BARRIER;
                    }
                } else {
                    if(rank < std::min(procNum, arrSize) - 1) {
                        int recvSize;
                        // printf("[Rank %d] get handleSize from %d\n", rank, rank + 1);
                        MPI_Recv(&recvSize, 1, MPI_INT, rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        float recvData[recvSize];
                        // printf("[Rank %d] get recvData from %d\n", rank, rank + 1);
                        MPI_Recv(recvData, recvSize, MPI_FLOAT, rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        if(data[handleSize - 1] > recvData[0]) {
                            isSorted = false;
                            float* merged = (float*)malloc((handleSize + recvSize) * sizeof(float));
                            memcpy(merged, data, handleSize * sizeof(float));
                            memcpy(merged + handleSize, recvData, recvSize * sizeof(float));
                            qsort(merged, handleSize + recvSize, sizeof(float), cmp);
                            memcpy(data, merged, handleSize * sizeof(float));
                            // printf("[Rank %d] send merged to %d\n", rank, rank + 1);
                            MPI_Send(merged + handleSize, recvSize, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
                            free(merged);
                        } else {
                            MPI_Send(recvData, recvSize, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
                        }
                    } else {
                        goto BARRIER;
                    }
                }
            } else {
                if(rank % 2 == 0) {
                    if(rank < std::min(procNum, arrSize) - 1) {
                        int recvSize;
                        // printf("[Rank %d] get recvSize from %d\n", rank, rank + 1);
                        MPI_Recv(&recvSize, 1, MPI_INT, rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        float recvData[recvSize];
                        // printf("[Rank %d] get recvData from %d\n", rank, rank + 1);
                        MPI_Recv(recvData ,recvSize , MPI_FLOAT, rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        if(data[handleSize - 1] > recvData[0]) {
                            isSorted = false;
                            float* merged = (float*)malloc((handleSize + recvSize) * sizeof(float));
                            memcpy(merged, data, handleSize * sizeof(float));
                            memcpy(merged + handleSize, recvData, recvSize * sizeof(float));
                            qsort(merged, handleSize + recvSize, sizeof(float), cmp);
                            memcpy(data, merged, handleSize * sizeof(float));
                            // printf("[Rank %d] send merged to %d\n", rank, rank + 1);
                            MPI_Send(merged + handleSize, recvSize, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
                            free(merged);
                        } else {
                            MPI_Send(recvData, recvSize, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
                        }
                    } else {
                        goto BARRIER;
                    }
                } else {
                    if(rank > 0) {
                        // printf("[Rank %d] send handleSize to %d\n", rank, rank - 1);
                        MPI_Send(&handleSize, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
                        // printf("[Rank %d] send data to %d\n", rank, rank - 1);
                        MPI_Send(data, handleSize, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD);
                        // printf("[Rank %d] recv data from %d\n", rank, rank - 1);
                        MPI_Recv(data, handleSize, MPI_FLOAT, rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    } else {
                        goto BARRIER;
                    }
                }
            }
            BARRIER:
            isOddStage = !isOddStage;
            MPI_Allreduce(&isSorted, &isSortedAll, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
        }
        // printf("[Rank %d] break loop\n", rank);

        // MPI write file
        MPI_File fout;
        MPI_File_open(MPI_COMM_WORLD, outFileName, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fout);
        MPI_File_write_at_all(fout, base * sizeof(float) + offset * sizeof(float), data, handleSize, MPI_FLOAT, MPI_STATUS_IGNORE);
        MPI_File_close(&fout);
        // printf("[Rank %d] write file done\n", rank);

        int newBase;
        MPI_Allreduce(&handleSize, &newBase, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        base += newBase;
        // if(rank == 0)
        //     printf("base = %d\n", base);
    }

    // // MPI read file
    // int handleSize = cutSize;
    // float data[2000000];
    // printf("[Rank %d] declare data done\n", rank);
    // MPI_File fin;
    // MPI_File_open(MPI_COMM_WORLD, inFileName, MPI_MODE_RDONLY, MPI_INFO_NULL, &fin);
    // MPI_File_read_at_all(fin, offset * sizeof(float), data, handleSize, MPI_FLOAT, MPI_STATUS_IGNORE);
    // MPI_File_close(&fin);
    // printf("[Rank %d] MPI read file done\n", rank);

    // // Initial sort
    // qsort(data, handleSize, sizeof(float), cmp);
    // MPI_Barrier(MPI_COMM_WORLD);
    // printf("[Rank %d] Initial sort done\n", rank);

    // // Odd-Even-Sort
    // bool isSortedAll = false;
    // bool isOddStage = true;
    // while(!isSortedAll) {
    //     if(isOddStage)
    //         printf("[Rank %d] Odd stage start\n", rank);
    //     else
    //         printf("[Rank %d] Even stage start\n", rank);
    //     bool isSorted = true;
    //     if(rank >= arrSize)
    //         goto BARRIER;
    //     if(isOddStage) {
    //         if(rank % 2 == 0) {
    //             if(rank > 0) {
    //                 printf("[Rank: %d] send handleSize to %d\n", rank, rank - 1);
    //                 MPI_Send(&handleSize, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
    //                 printf("[Rank %d] send data to %d\n", rank, rank - 1);
    //                 MPI_Send(data , handleSize , MPI_FLOAT , rank - 1 , 0, MPI_COMM_WORLD);
    //                 printf("[Rank %d] get handleSize from %d\n", rank, rank - 1);
    //                 MPI_Recv(data, handleSize, MPI_FLOAT, rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //             } else {
    //                 goto BARRIER;
    //             }
    //         } else {
    //             if(rank < std::min(procNum, arrSize) - 1) {
    //                 int recvSize;
    //                 printf("[Rank %d] get handleSize from %d\n", rank, rank + 1);
    //                 MPI_Recv(&recvSize, 1, MPI_INT, rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //                 float recvData[recvSize];
    //                 printf("[Rank %d] get recvData from %d\n", rank, rank + 1);
    //                 MPI_Recv(recvData, recvSize, MPI_FLOAT, rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //                 if(data[handleSize - 1] > recvData[0]) {
    //                     isSorted = false;
    //                     float* merged = (float*)malloc((handleSize + recvSize) * sizeof(float));
    //                     memcpy(merged, data, handleSize * sizeof(float));
    //                     memcpy(merged + handleSize, recvData, recvSize * sizeof(float));
    //                     qsort(merged, handleSize + recvSize, sizeof(float), cmp);
    //                     memcpy(data, merged, handleSize * sizeof(float));
    //                     printf("[Rank %d] send merged to %d\n", rank, rank + 1);
    //                     MPI_Send(merged + handleSize, recvSize, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
    //                     free(merged);
    //                 } else {
    //                     MPI_Send(recvData, recvSize, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
    //                 }
    //             } else {
    //                 goto BARRIER;
    //             }
    //         }
    //     } else {
    //         if(rank % 2 == 0) {
    //             if(rank < std::min(procNum, arrSize) - 1) {
    //                 int recvSize;
    //                 printf("[Rank %d] get recvSize from %d\n", rank, rank + 1);
    //                 MPI_Recv(&recvSize, 1, MPI_INT, rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //                 float recvData[recvSize];
    //                 printf("[Rank %d] get recvData from %d\n", rank, rank + 1);
    //                 MPI_Recv(recvData ,recvSize , MPI_FLOAT, rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //                 if(data[handleSize - 1] > recvData[0]) {
    //                     isSorted = false;
    //                     float* merged = (float*)malloc((handleSize + recvSize) * sizeof(float));
    //                     memcpy(merged, data, handleSize * sizeof(float));
    //                     memcpy(merged + handleSize, recvData, recvSize * sizeof(float));
    //                     qsort(merged, handleSize + recvSize, sizeof(float), cmp);
    //                     memcpy(data, merged, handleSize * sizeof(float));
    //                     printf("[Rank %d] send merged to %d\n", rank, rank + 1);
    //                     MPI_Send(merged + handleSize, recvSize, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
    //                     free(merged);
    //                 } else {
    //                     MPI_Send(recvData, recvSize, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
    //                 }
    //             } else {
    //                 goto BARRIER;
    //             }
    //         } else {
    //             if(rank > 0) {
    //                 printf("[Rank %d] send handleSize to %d\n", rank, rank - 1);
    //                 MPI_Send(&handleSize, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
    //                 printf("[Rank %d] send data to %d\n", rank, rank - 1);
    //                 MPI_Send(data, handleSize, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD);
    //                 printf("[Rank %d] recv data from %d\n", rank, rank - 1);
    //                 MPI_Recv(data, handleSize, MPI_FLOAT, rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //             } else {
    //                 goto BARRIER;
    //             }
    //         }
    //     }
    //     BARRIER:
    //     isOddStage = !isOddStage;
    //     MPI_Allreduce(&isSorted, &isSortedAll, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
    // }
    // printf("[Rank %d] break loop\n", rank);

    // // MPI write file
    // MPI_File fout;
    // MPI_File_open(MPI_COMM_WORLD, outFileName, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fout);
    // MPI_File_write_at_all(fout, offset * sizeof(float), data, handleSize, MPI_FLOAT, MPI_STATUS_IGNORE);
    // MPI_File_close(&fout);
    
    // MPI finalize
    MPI_Finalize();
}
