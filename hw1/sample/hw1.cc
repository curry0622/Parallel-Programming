#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <cstdio>
#include <mpi.h>

int cmp (const void * a, const void * b) {
   float fa = *(float *)a;
   float fb = *(float *)b;
   return (fa > fb) ? 1 : -1;
}

int main(int argc, char** argv) {
    // MPI init
    int rank, procNum;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);

    // Get args
    int arrSize = atoi(argv[1]);
    char* inFileName = argv[2];
    char* outFileName = argv[3];

    // Set handle size
    int handleSize = 0, avgSize = 0;
    if(arrSize % procNum != 0) {
        avgSize = arrSize / procNum + 1;
        if(rank != procNum - 1)
            handleSize = arrSize / procNum + 1;
        else
            handleSize = arrSize - (procNum - 1) * (arrSize / procNum + 1);
    } else {
        avgSize = arrSize / procNum;
        handleSize = arrSize / procNum;
    }

    // MPI read file
    float data[handleSize];
    MPI_File fin;
    MPI_File_open(MPI_COMM_WORLD, inFileName, MPI_MODE_RDONLY, MPI_INFO_NULL, &fin);
    MPI_File_read_at_all(fin, sizeof(float) * rank * avgSize, data, handleSize, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&fin);

    // Initial sort
    qsort(data, handleSize, sizeof(float), cmp);
    MPI_Barrier(MPI_COMM_WORLD);

    // // Create odd & even groups
    // MPI_Group worldGroup, oddGroup, evenGroup;
    // MPI_Comm oddComm, evenComm;
    // MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);

    // int oddGroupSize, evenGroupSize;
    // if(rank % 2 == 0) {
    //     evenGroupSize = rank == procNum - 1 ? 1 : 2;
    //     oddGroupSize = rank == 0 ? 1 : 2;
    // } else {
    //     evenGroupSize = rank == 0 ? 1 : 2;
    //     oddGroupSize = rank == procNum - 1 ? 1 : 2;
    // }

    // int oddGroupRanks[oddGroupSize], evenGroupRanks[evenGroupSize];
    // if(rank % 2 == 0) {
    //     if(rank == procNum - 1) {
    //         evenGroupRanks[0] = rank;
    //     } else {
    //         evenGroupRanks[0] = rank;
    //         evenGroupRanks[1] = rank + 1;
    //     }
    //     if(rank == 0) {
    //         oddGroupRanks[0] = rank;
    //     } else {
    //         oddGroupRanks[0] = rank - 1;
    //         oddGroupRanks[1] = rank;
    //     }
    // } else {
    //     if(rank == 0) {
    //         evenGroupRanks[0] = rank;
    //     } else {
    //         evenGroupRanks[0] = rank - 1;
    //         evenGroupRanks[1] = rank;
    //     }
    //     if(rank == procNum - 1) {
    //         oddGroupRanks[0] = rank;
    //     } else {
    //         oddGroupRanks[0] = rank;
    //         oddGroupRanks[1] = rank + 1;
    //     }
    // }

    // MPI_Group_incl(worldGroup, oddGroupSize, oddGroupRanks, &oddGroup);
    // MPI_Group_incl(worldGroup, evenGroupSize, evenGroupRanks, &evenGroup);
    // MPI_Comm_create(MPI_COMM_WORLD, oddGroup, &oddComm);
    // MPI_Comm_create(MPI_COMM_WORLD, evenGroup, &evenComm);

    // Odd-Even-Sort
    bool isSortedAll = false;
    bool isOddStage = true;
    while(!isSortedAll) {
        if(isOddStage)
            printf("[][][Rank: %d, Odd stage start][][]\n", rank);
        else
            printf("[][][Rank: %d, Even stage start][][]\n", rank);
        bool isSorted = true;
        if(isOddStage) {
            if(rank % 2 == 0) {
                if(rank > 0) {
                    printf("Rank: %d, send handleSize to %d\n", rank, rank - 1);
                    MPI_Send(&handleSize, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
                    printf("Rank: %d, send data to %d\n", rank, rank - 1);
                    MPI_Send(data , handleSize , MPI_FLOAT , rank - 1 , 0, MPI_COMM_WORLD);
                    printf("Rank: %d, get handleSize from %d\n", rank, rank - 1);
                    MPI_Recv(data, handleSize, MPI_FLOAT, rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } else {
                    goto BARRIER;
                }
            } else {
                if(rank < procNum - 1) {
                    int recvSize;
                    printf("Rank: %d, get handleSize from %d\n", rank, rank + 1);
                    MPI_Recv(&recvSize, 1, MPI_INT, rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    float recvData[recvSize];
                    printf("Rank: %d, get recvData from %d\n", rank, rank + 1);
                    MPI_Recv(recvData, recvSize, MPI_FLOAT, rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if(data[handleSize - 1] > recvData[0]) {
                        isSorted = false;
                        float* merged = (float*)malloc((handleSize + recvSize) * sizeof(float));
                        memcpy(merged, data, handleSize * sizeof(float));
                        memcpy(merged + handleSize, recvData, recvSize * sizeof(float));
                        qsort(merged, handleSize + recvSize, sizeof(float), cmp);
                        memcpy(data, merged, handleSize * sizeof(float));
                        printf("Rank: %d, send merged to %d\n", rank, rank + 1);
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
                if(rank < procNum - 1) {
                    int recvSize;
                    printf("Rank: %d, get recvSize from %d\n", rank, rank + 1);
                    MPI_Recv(&recvSize, 1, MPI_INT, rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    float recvData[recvSize];
                    printf("Rank: %d, get recvData from %d\n", rank, rank + 1);
                    MPI_Recv(recvData ,recvSize , MPI_FLOAT, rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if(data[handleSize - 1] > recvData[0]) {
                        isSorted = false;
                        float* merged = (float*)malloc((handleSize + recvSize) * sizeof(float));
                        memcpy(merged, data, handleSize * sizeof(float));
                        memcpy(merged + handleSize, recvData, recvSize * sizeof(float));
                        qsort(merged, handleSize + recvSize, sizeof(float), cmp);
                        memcpy(data, merged, handleSize * sizeof(float));
                        printf("Rank: %d, send merged to %d\n", rank, rank + 1);
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
                    printf("Rank: %d, send handleSize to %d\n", rank, rank - 1);
                    MPI_Send(&handleSize, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
                    printf("Rank: %d, send data to %d\n", rank, rank - 1);
                    MPI_Send(data, handleSize, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD);
                    printf("Rank: %d, recv data from %d\n", rank, rank - 1);
                    MPI_Recv(data, handleSize, MPI_FLOAT, rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } else {
                    goto BARRIER;
                }
            }
        }
        BARRIER:
        
        // if(isOddStage)
        //     printf("[][][Rank: %d, Odd stage finish][][]\n", rank);
        // else
        //     printf("[][][Rank: %d, Even stage finish][][]\n", rank);
        // for(int p = 0; p < procNum; p++) {
        //     if(rank == p) {
        //         printf("Rank: %d\n", rank);
        //         for(int i = 0; i < handleSize; i++) {
        //             printf("%f ", data[i]);
        //         }
        //         printf("\n");
        //     }
        // }
        isOddStage = !isOddStage;
        MPI_Allreduce(&isSorted, &isSortedAll, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
    }
    
    printf("Rank: %d, break loop\n", rank);

    // MPI wriite file
    MPI_File fout;
    MPI_File_open(MPI_COMM_WORLD, outFileName, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fout);
    MPI_File_write_at_all(fout, sizeof(float) * rank * avgSize, data, handleSize, MPI_FLOAT, MPI_STATUS_IGNORE);
    // MPI_File_read_at_all(fin, sizeof(float) * rank * avgSize, data, handleSize, MPI_FLOAT, MPI_STATUS_IGNORE);
    // MPI_File_close(&fin);
    
    // MPI finalize
    MPI_Finalize();
}
