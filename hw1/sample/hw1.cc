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
    char inFileName[] = argv[2];
    char outFileName[] = argv[3];

    printf("array size: %d\n", arrSize);
    printf("input file name: %s\n", inFileName);
    printf("output file name: %s\n", outFileName);

    // MPI finalize
    MPI_Finalize();
}
