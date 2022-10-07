#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <mpi.h>

int cmp(const void * a, const void * b) {
   float fa = *(float *)a;
   float fb = *(float *)b;
   return (fa > fb) ? 1 : -1;
}

int getOffset(int rank, int arrSize, int procNum) {
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
    double startTime = MPI_Wtime();

    // Get args
    int arrSize = atoi(argv[1]);
    char* inFileName = argv[2];
    char* outFileName = argv[3];

    // Set handle size
    int handleSize = arrSize / procNum;
    int left = arrSize % procNum;
    if (rank < left)
        handleSize++;
    int offset = getOffset(rank, arrSize, procNum);

    // Allocate memory
    float* data = (float*)malloc(sizeof(float) * (handleSize + 1) * 2);

    // MPI read file
    MPI_File fin;
    MPI_File_open(MPI_COMM_WORLD, inFileName, MPI_MODE_RDONLY, MPI_INFO_NULL, &fin);
    MPI_File_read_at_all(fin, offset * sizeof(float), data, handleSize, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&fin);

    // Initial sort
    qsort(data, handleSize, sizeof(float), cmp);
    MPI_Barrier(MPI_COMM_WORLD);

    // Odd-Even-Sort
    bool isSortedAll = false;
    bool isOddStage = true;
    while(!isSortedAll) {
        bool isSorted = true;
        if(rank >= arrSize)
            goto BARRIER;
        if(isOddStage) {
            if(rank % 2 == 0) {
                if(rank > 0) {
                    MPI_Send(&handleSize, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
                    MPI_Send(data, handleSize, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD);
                    MPI_Recv(data, handleSize, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            } else {
                if(rank < std::min(procNum, arrSize) - 1) {
                    int recvSize;
                    MPI_Recv(&recvSize, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(data + handleSize, recvSize, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if(data[handleSize - 1] > data[handleSize]) {
                        isSorted = false;
                        qsort(data, handleSize + recvSize, sizeof(float), cmp);
                        MPI_Send(data + handleSize, recvSize, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
                    } else {
                        MPI_Send(data + handleSize, recvSize, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
                    }
                }
            }
        } else {
            if(rank % 2 == 0) {
                if(rank < std::min(procNum, arrSize) - 1) {
                    int recvSize;
                    MPI_Recv(&recvSize, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(data + handleSize ,recvSize , MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if(data[handleSize - 1] > data[handleSize]) {
                        isSorted = false;
                        qsort(data, handleSize + recvSize, sizeof(float), cmp);
                        MPI_Send(data + handleSize, recvSize, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
                    } else {
                        MPI_Send(data + handleSize, recvSize, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
                    }
                }
            } else {
                if(rank > 0) {
                    MPI_Send(&handleSize, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
                    MPI_Send(data, handleSize, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD);
                    MPI_Recv(data, handleSize, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        }
        BARRIER:
        isOddStage = !isOddStage;
        MPI_Allreduce(&isSorted, &isSortedAll, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
    }

    // MPI write file
    MPI_File fout;
    MPI_File_open(MPI_COMM_WORLD, outFileName, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fout);
    MPI_File_write_at_all(fout, offset * sizeof(float), data, handleSize, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&fout);
    
    // MPI finalize
    if(rank == 0) {
        printf("Time: %f\n", MPI_Wtime() - startTime);
    }
    MPI_Finalize();
}
