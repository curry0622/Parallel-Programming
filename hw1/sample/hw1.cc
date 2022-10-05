#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <mpi.h>

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
        if(rank != procNum - 1)
            handleSize = arrSize / procNum + 1;
        else
            handleSize = arrSize - (procNum - 1) * (arrSize / procNum + 1);
        avgSize = arrSize / procNum + 1;
    } else {
        handleSize = arrSize / procNum;
        avgSize = arrSize / procNum;
    }

    // MPI read file
    float data[handleSize];
    MPI_File fin;
    MPI_File_open(MPI_COMM_WORLD, inFileName, MPI_MODE_RDONLY, MPI_INFO_NULL, &fin);
    MPI_File_read_at_all(fin, sizeof(float) * rank * avgSize, data, handleSize, MPI_FLOAT, MPI_STATUS_IGNORE);

    // MPI finalize
    MPI_Finalize();
}
