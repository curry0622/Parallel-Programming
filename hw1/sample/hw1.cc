#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <cstdio>
#include <cfloat>
#include <algorithm>
#include <mpi.h>
#include <iostream>
#include <boost/sort/spreadsort/spreadsort.hpp>

#define DEBUG false

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

int getHalfSize(float* arr, int size, float std) {
    int low = 0, high = size;
    while(low != high) {
        int mid = (low + high) / 2;
        if(arr[mid] <= std)
            low = mid + 1;
        else
            high = mid;
    }
    return low;
}

void mergeData(float* myDataBuf, int mySize, float* recvDataBuf, int recvSize, bool fromMin, float* funcBuf) {
    float* newDataBuf = funcBuf;
    if(fromMin) {
        int i = 0, j = 0;
        while(i < mySize && j < recvSize && i + j < mySize) {
            if(myDataBuf[i] < recvDataBuf[j]) {
                newDataBuf[i + j] = myDataBuf[i];
                i++;
            } else {
                newDataBuf[i + j] = recvDataBuf[j];
                j++;
            }
        }
        while (i < mySize && i + j < mySize) {
            newDataBuf[i + j] = myDataBuf[i];
            i++;
        }
        while (j < recvSize && i + j < mySize) {
            newDataBuf[i + j] = recvDataBuf[j];
            j++;
        }
    } else {
        int i = mySize - 1, j = recvSize - 1;
        int idx = mySize - 1;
        while(i >= 0 && j >= 0 && idx >= 0) {
            if(myDataBuf[i] > recvDataBuf[j]) {
                newDataBuf[idx] = myDataBuf[i];
                i--;
            } else {
                newDataBuf[idx] = recvDataBuf[j];
                j--;
            }
            idx--;
        }
        while(i >= 0 && idx >= 0) {
            newDataBuf[idx] = myDataBuf[i];
            i--;
            idx--;
        }
        while(j >= 0 && idx >= 0) {
            newDataBuf[idx] = recvDataBuf[j];
            j--;
            idx--;
        }
    }
    memcpy(myDataBuf, newDataBuf, sizeof(float) * mySize);
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

    // Exclude processes when arrSize < size
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

    // Set mySize
    int mySize = arrSize / size;
    int left = arrSize % size;
    if (rank < left)
        mySize++;
    MPI_Offset offset = getOffset(rank, size, arrSize) * sizeof(float);

    // Allocate memory
    float* myDataBuf = (float*)malloc(sizeof(float) * (mySize + 1));
    float* recvDataBuf = (float*)malloc(sizeof(float) * (mySize + 1));
    float* funcBuf = (float*)malloc(sizeof(float) * (mySize + 1));

    // MPI read file
    MPI_File fin;
    MPI_File_open(comm, inFileName, MPI_MODE_RDONLY, MPI_INFO_NULL, &fin);
    MPI_File_read_at(fin, offset, myDataBuf, mySize, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&fin);

    // Initial sort
    boost::sort::spreadsort::float_sort(myDataBuf, myDataBuf + mySize);
    MPI_Barrier(comm);

    // Odd even sort
    if(size > 1) {
        bool isSortedAll = false;
        bool isOddPhase = true;
        while(!isSortedAll) {
            bool isSorted = true;

            if(DEBUG) {
                if(isOddPhase) std::cout << rank << " odd" << std::endl;
                else std::cout << rank << " even" << std::endl;
            }

            // Edge cases
            if(isOddPhase) {
                // rank[0] recv from rank[1]
                if(rank == 0) {
                    float recvData;
                    if(DEBUG) std::cout << "Rank " << 0 << ", req recv from " << 1 << std::endl;
                    MPI_Recv(&recvData, 1, MPI_FLOAT, 1, 0, comm, MPI_STATUS_IGNORE);
                    if(DEBUG) std::cout << "Rank " << 0 << ", fin recv from " << 1 << std::endl;
                    if(recvData < myDataBuf[mySize - 1]) {
                        isSorted = false;
                    }
                    goto LABEL_BARRIER;
                }
                // rank[1] send to rank[0]
                if(rank == 1) {
                    if(DEBUG) std::cout << "Rank " << 1 << ", req send to " << 0 << std::endl;
                    MPI_Send(myDataBuf, 1, MPI_FLOAT, 0, 0, comm);
                    if(DEBUG) std::cout << "Rank " << 1 << ", fin send to " << 0 << std::endl;
                }
                if(size % 2 == 0 && size > 2) {
                    // rank[size - 1] recv from rank[size - 2]
                    if(rank == size - 1) {
                        float recvData;
                        if(DEBUG) std::cout << "Rank " << size - 1 << ", req recv from " << size - 2 << std::endl;
                        MPI_Recv(&recvData, 1, MPI_FLOAT, size - 2, 0, comm, MPI_STATUS_IGNORE);
                        if(DEBUG) std::cout << "Rank " << size - 1 << ", fin recv from " << size - 2 << std::endl;
                        if(recvData > myDataBuf[0]) {
                            isSorted = false;
                        }
                        goto LABEL_BARRIER;
                    }
                    // rank[size - 2] send to rank[size - 1]
                    if(rank == size - 2) {
                        if(DEBUG) std::cout << "Rank " << size - 2 << ", req send to " << size - 1 << std::endl;
                        MPI_Send(myDataBuf + mySize - 1, 1, MPI_FLOAT, size - 1, 0, comm);
                        if(DEBUG) std::cout << "Rank " << size - 2 << ", fin send to " << size - 1 << std::endl;
                    }
                }
            } else if(size % 2 == 1) {
                // rank[size - 1] recv from rank[size - 2]
                if(rank == size - 1) {
                    float recvData;
                    if(DEBUG) std::cout << "Rank " << size - 1 << ", req recv from " << size - 2 << std::endl;
                    MPI_Recv(&recvData, 1, MPI_FLOAT, size - 2, 0, comm, MPI_STATUS_IGNORE);
                    if(DEBUG) std::cout << "Rank " << size - 1 << ", fin recv from " << size - 2 << std::endl;
                    if(recvData > myDataBuf[0]) {
                        isSorted = false;
                    }
                    goto LABEL_BARRIER;
                }
                // rank[size - 2] send to rank[size - 1]
                if(rank == size - 2) {
                    if(DEBUG) std::cout << "Rank " << size - 2 << ", req send to " << size - 1 << std::endl;
                    MPI_Send(myDataBuf + mySize - 1, 1, MPI_FLOAT, size - 1, 0, comm);
                    if(DEBUG) std::cout << "Rank " << size - 2 << ", fin send to " << size - 1 << std::endl;
                }
            }

            // Normal cases
            if((isOddPhase && rank % 2 == 0) || (!isOddPhase && rank % 2 == 1)) {
                // Send and recv median
                float recvMax;
                if(DEBUG) std::cout << "A Rank " << rank << ", req send recv " << rank - 1 << std::endl;
                MPI_Sendrecv(
                    myDataBuf, 1, MPI_FLOAT, rank - 1, 0,
                    &recvMax, 1, MPI_FLOAT, rank - 1, 0,
                    comm, MPI_STATUS_IGNORE
                );
                if(DEBUG) std::cout << "A Rank " << rank << ", fin send recv " << rank - 1 << std::endl;
                if(recvMax > myDataBuf[0]) {
                    isSorted = false;
                    // Send and recv size
                    int sendSize = getHalfSize(myDataBuf, mySize, recvMax);
                    int recvSize;
                    if(DEBUG) std::cout << "B Rank " << rank << ", req send recv " << rank - 1 << std::endl;
                    MPI_Sendrecv(
                        &sendSize, 1, MPI_INT, rank - 1, 0,
                        &recvSize, 1, MPI_INT, rank - 1, 0,
                        comm, MPI_STATUS_IGNORE
                    );
                    if(DEBUG) std::cout << "B Rank " << rank << ", fin send recv " << rank - 1 << std::endl;
                    // Send and recv data
                    if(DEBUG) std::cout << "C Rank " << rank << ", req send recv " << rank - 1 << std::endl;
                    MPI_Sendrecv(
                        myDataBuf, sendSize, MPI_FLOAT, rank - 1, 0,
                        recvDataBuf, recvSize, MPI_FLOAT, rank - 1, 0,
                        comm, MPI_STATUS_IGNORE
                    );
                    if(DEBUG) std::cout << "C Rank " << rank << ", fin send recv " << rank - 1 << std::endl;
                    // Merge data
                    mergeData(myDataBuf, mySize, recvDataBuf, recvSize, false, funcBuf);
                }
            } else {
                // Send and recv median
                float recvMin;
                if(DEBUG) std::cout << "D Rank " << rank << ", req send recv " << rank + 1 << std::endl;
                MPI_Sendrecv(
                    myDataBuf + mySize - 1, 1, MPI_FLOAT, rank + 1, 0,
                    &recvMin, 1, MPI_FLOAT, rank + 1, 0,
                    comm, MPI_STATUS_IGNORE
                );
                if(DEBUG) std::cout << "D Rank " << rank << ", fin send recv " << rank + 1 << std::endl;
                if(recvMin < myDataBuf[mySize - 1]) {
                    isSorted = false;
                    // Send and recv size
                    int sendSize = mySize - getHalfSize(myDataBuf, mySize, recvMin);
                    int recvSize;
                    if(DEBUG) std::cout << "E Rank " << rank << ", req send recv " << rank + 1 << std::endl;
                    MPI_Sendrecv(
                        &sendSize, 1, MPI_INT, rank + 1, 0,
                        &recvSize, 1, MPI_INT, rank + 1, 0,
                        comm, MPI_STATUS_IGNORE
                    );
                    if(DEBUG) std::cout << "E Rank " << rank << ", fin send recv " << rank + 1 << std::endl;
                    // Send and recv data
                    if(DEBUG) std::cout << "F Rank " << rank << ", req send recv " << rank + 1 << std::endl;
                    MPI_Sendrecv(
                        myDataBuf + mySize - sendSize, sendSize, MPI_FLOAT, rank + 1, 0,
                        recvDataBuf, recvSize, MPI_FLOAT, rank + 1, 0,
                        comm, MPI_STATUS_IGNORE
                    );
                    if(DEBUG) std::cout << "F Rank " << rank << ", fin send recv " << rank + 1 << std::endl;
                    // Merge data
                    mergeData(myDataBuf, mySize, recvDataBuf, recvSize, true, funcBuf);
                }
            }

            LABEL_BARRIER:
            isOddPhase = !isOddPhase;
            MPI_Allreduce(&isSorted, &isSortedAll, 1, MPI_C_BOOL, MPI_LAND, comm);
        }
    }

    // MPI write file
    MPI_File fout;
    MPI_File_open(comm, outFileName, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fout);
    MPI_File_write_at(fout, offset, myDataBuf, mySize, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&fout);
    
    // MPI finalize
    free(myDataBuf);
    MPI_Finalize();
}
