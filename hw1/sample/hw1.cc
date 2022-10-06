#include <stdio.h>
#include <stdlib.h>
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

    // Initial sort
    qsort(data, handleSize, sizeof(float), cmp);
    MPI_Barrier(MPI_COMM_WORLD);

    // Create odd & even groups
    MPI_Group worldGroup, oddGroup, evenGroup;
    MPI_Comm oddComm, evenComm;
    MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);
    int oddGroupSize, evenGroupSize;
    if(rank % 2 == 0) {
        evenGroupSize = rank == procNum - 1 ? 1 : 2;
        oddGroupSize = rank == 0 ? 1 : 2;
    } else {
        evenGroupSize = rank == 0 ? 1 : 2;
        oddGroupSize = rank == procNum - 1 ? 1 : 2;
    }
    int oddGroupRanks[oddGroupSize], evenGroupRanks[evenGroupSize];
    if(rank % 2 == 0) {
        if(rank == procNum - 1) {
            evenGroupRanks[0] = rank;
        } else {
            evenGroupRanks[0] = rank;
            evenGroupRanks[1] = rank + 1;
        }
        if(rank == 0) {
            oddGroupRanks[0] = rank;
        } else {
            oddGroupRanks[0] = rank - 1;
            oddGroupRanks[1] = rank;
        }
    } else {
        if(rank == 0) {
            evenGroupRanks[0] = rank;
        } else {
            evenGroupRanks[0] = rank - 1;
            evenGroupRanks[1] = rank;
        }
        if(rank == procNum - 1) {
            oddGroupRanks[0] = rank;
        } else {
            oddGroupRanks[0] = rank;
            oddGroupRanks[1] = rank + 1;
        }
    }
    MPI_Group_incl(worldGroup, oddGroupSize, oddGroupRanks, &oddGroup);
    MPI_Group_incl(worldGroup, evenGroupSize, evenGroupRanks, &evenGroup);
    MPI_Comm_create(MPI_COMM_WORLD, oddGroup, &oddComm);
    MPI_Comm_create(MPI_COMM_WORLD, evenGroup, &evenComm);

    // Odd-Even-Sort
    // bool isSorted = false;
    // while (isSorted) {
    //     // odd
    //     for(int i = 1; i <= procNum - 2; i += 2) {
    //         if() {

    //         }
    //     }
    // }
    

    // MPI finalize
    MPI_Finalize();
}
