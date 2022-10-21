#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <cstdio>
#include <cfloat>
#include <algorithm>
#include <mpi.h>
// #include <boost/sort/spreadsort/spreadsort.hpp>
// #include <boost/sort/sort.hpp>
#include <boost/sort/spreadsort/float_sort.hpp>
// #include <boost/sort/pdqsort/pdqsort.hpp>

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

void mergeSort(float* arr1, int num1, float* arr2, int num2) {
    int i = 0, j = 0;
    float* merged = (float*)malloc(sizeof(float) * (num1 + num2));
    while (i < num1 && j < num2) {
        if (arr1[i] < arr2[j]) {
            merged[i + j] = arr1[i];
            i++;
        } else {
            merged[i + j] = arr2[j];
            j++;
        }
    }
    while (i < num1) {
        merged[i + j] = arr1[i];
        i++;
    }
    while (j < num2) {
        merged[i + j] = arr2[j];
        j++;
    }
    memcpy(arr1, merged, sizeof(float) * num1);
    memcpy(arr2, merged + num1, sizeof(float) * num2);
    free(merged);
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

    // Calculate last rank
    int lastRank = std::min(procNum, arrSize) - 1;

    // Allocate memory
    float* myDataBuf = (float*)malloc(sizeof(float) * (handleSize + 1) * 2);
    float* adjDataBuf = (float*)malloc(sizeof(float) * (handleSize + 1));

    // MPI read file
    MPI_File fin;
    MPI_File_open(MPI_COMM_WORLD, inFileName, MPI_MODE_RDONLY, MPI_INFO_NULL, &fin);
    MPI_File_read_at_all(fin, offset * sizeof(float), myDataBuf, handleSize, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&fin);

    // Initial sort
    // qsort(myDataBuf, handleSize, sizeof(float), cmp);
    // boost::sort::spreadsort::spreadsort(myDataBuf, myDataBuf + handleSize);
    // boost::sort::parallel_stable_sort(myDataBuf, myDataBuf + handleSize);
    boost::sort::spreadsort::float_sort(myDataBuf, myDataBuf + handleSize);
    // boost::sort::pdqsort(myDataBuf, myDataBuf + handleSize);
    MPI_Barrier(MPI_COMM_WORLD);

    // Odd-Even-Sort
    bool isSortedAll = false;
    bool isOddStage = true;
    while(!isSortedAll) {
        bool isSorted = true;
        if(rank >= arrSize)
            goto BARRIER;
        if(isOddStage) {
            // Odd stage
            if(rank % 2 == 0) {
                if(rank > 0) {
                    MPI_Send(&handleSize, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
                    MPI_Send(myDataBuf, handleSize, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD);
                    MPI_Recv(myDataBuf, handleSize, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if(rank == lastRank - 1 && lastRank % 2 == 1) {
                        MPI_Send(myDataBuf + handleSize - 1, 1, MPI_FLOAT, lastRank, 0, MPI_COMM_WORLD);
                    }
                } else if(lastRank >= 1) {
                    // Rank == 0
                    float nextData;
                    MPI_Recv(&nextData, 1, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if(nextData < myDataBuf[handleSize - 1])
                        isSorted = false;
                }
            } else {
                if(rank < lastRank) {
                    int recvSize;
                    MPI_Recv(&recvSize, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(adjDataBuf, recvSize, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if(myDataBuf[handleSize - 1] > adjDataBuf[0]) {
                        isSorted = false;
                        mergeSort(myDataBuf, handleSize, adjDataBuf, recvSize);
                        MPI_Send(adjDataBuf, recvSize, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
                    } else {
                        MPI_Send(adjDataBuf, recvSize, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
                    }
                    if(rank == 1) {
                        MPI_Send(myDataBuf, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
                    }
                } else if(lastRank >= 1 && lastRank % 2 == 1) {
                    float lastData;
                    MPI_Recv(&lastData, 1, MPI_FLOAT, lastRank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if(lastData > myDataBuf[0]) {
                        isSorted = false;
                    }
                }
            }
        } else {
            // Even stage
            if(rank % 2 == 0) {
                if(rank < lastRank) {
                    int recvSize;
                    MPI_Recv(&recvSize, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(adjDataBuf, recvSize, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if(myDataBuf[handleSize - 1] > adjDataBuf[0]) {
                        isSorted = false;
                        mergeSort(myDataBuf, handleSize, adjDataBuf, recvSize);
                        MPI_Send(adjDataBuf, recvSize, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
                    } else {
                        MPI_Send(adjDataBuf, recvSize, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
                    }
                } else if(lastRank >= 1 && lastRank % 2 == 0) {
                    // Rank == lastRank
                    float lastData;
                    MPI_Recv(&lastData, 1, MPI_FLOAT, lastRank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if(lastData > myDataBuf[0]) {
                        isSorted = false;
                    }
                }
            } else {
                if(rank > 0) {
                    MPI_Send(&handleSize, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
                    MPI_Send(myDataBuf, handleSize, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD);
                    MPI_Recv(myDataBuf, handleSize, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if(lastRank >= 1 && lastRank % 2 == 0 && rank == lastRank - 1) {
                        MPI_Send(myDataBuf + handleSize - 1, 1, MPI_FLOAT, lastRank, 0, MPI_COMM_WORLD);
                    }
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
    MPI_File_write_at_all(fout, offset * sizeof(float), myDataBuf, handleSize, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&fout);
    
    // MPI finalize
    free(myDataBuf);
    MPI_Finalize();
}
