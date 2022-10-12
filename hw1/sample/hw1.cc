#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <cstdio>
#include <cfloat>
#include <algorithm>
#include <mpi.h>
#include <boost/sort/spreadsort/spreadsort.hpp>

int getOffset(int rank, int size, int arrSize) {
    int offset = 0;
    int left = arrSize % size;
    int base = arrSize / size;
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
    // Get args
    const int arrSize = atoi(argv[1]);
    const char* inFileName = argv[2];
    const char* outFileName = argv[3];

    // MPI init
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Exclude process if needed
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Group orgGroup, newGroup;
    if(arrSize < size) {
        MPI_Comm_group(comm, &orgGroup);
        int ranges[][3] = {{arrSize, size - 1, 1}};
        MPI_Group_range_excl(orgGroup, 1, ranges, &newGroup);
        MPI_Comm_create(comm, newGroup, &comm);
        size = arrSize;
        if(comm == MPI_COMM_NULL) {
            MPI_Finalize();
            return 0;
        }
    }

    // Set handle size
    int handleSize = arrSize / size;
    int left = arrSize % size;
    if (rank < left)
        handleSize++;
    MPI_Offset offset = getOffset(rank, size, arrSize) * sizeof(float);

    // Allocate memory
    float* myDataBuf = (float*)malloc(sizeof(float) * (handleSize + 1) * 2);
    float* adjDataBuf = (float*)malloc(sizeof(float) * (handleSize + 1));

    // MPI read file
    MPI_File fin;
    MPI_File_open(comm, inFileName, MPI_MODE_RDONLY, MPI_INFO_NULL, &fin);
    MPI_File_read_at_all(fin, offset, myDataBuf, handleSize, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&fin);

    // Initial sort
    boost::sort::spreadsort::float_sort(myDataBuf, myDataBuf + handleSize);
    MPI_Barrier(comm);
    if(size == 1)
        goto LABEL_WRITE;

    // Odd even sort
    bool isSortedAll = false;
    bool isOddPhase = true;
    while(!isSortedAll) {
        bool isSorted = true;
        if(isOddPhase) {
            // Odd phase
            if(rank % 2 == 0) {
                
            } else {
                
            }
        } else {
            // Even phase
            if(rank % 2 == 0) {
                
            } else {
                
            }
        }
        isOddPhase = !isOddPhase;
        MPI_Allreduce(&isSorted, &isSortedAll, 1, MPI_C_BOOL, MPI_LAND, comm);
    }

    // MPI write file
    LABEL_WRITE:
    MPI_File fout;
    MPI_File_open(comm, outFileName, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fout);
    MPI_File_write_at_all(fout, offset, myDataBuf, handleSize, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&fout);
    
    // MPI finalize
    free(myDataBuf);
    MPI_Finalize();
}
