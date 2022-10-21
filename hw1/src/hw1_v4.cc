#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <cstdio>
#include <cfloat>
#include <algorithm>
#include <mpi.h>
#include <iostream>
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
    // MPI init
    int rank, size;
    MPI_Init(&argc, &argv);
    double timeIO = 0, timeComm = 0, timeCPU = 0, timeTmp = 0, timeSort = 0, timeMerge = 0, timeStart = MPI_Wtime();
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get args
    const int arrSize = atoi(argv[1]);
    const char* inFileName = argv[2];
    const char* outFileName = argv[3];

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
    timeTmp = MPI_Wtime();
    MPI_File fin;
    MPI_File_open(comm, inFileName, MPI_MODE_RDONLY, MPI_INFO_NULL, &fin);
    MPI_File_read_at_all(fin, offset, myDataBuf, mySize, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&fin);
    timeIO += MPI_Wtime() - timeTmp;

    // Initial sort
    timeTmp = MPI_Wtime();
    boost::sort::spreadsort::float_sort(myDataBuf, myDataBuf + mySize);
    timeSort += MPI_Wtime() - timeTmp;
    MPI_Barrier(comm);

    // Odd even sort
    if(size > 1) {
        bool isSortedAll = false;
        bool isOddPhase = true;
        while(!isSortedAll) {
            bool isSorted = true;

            // Edge cases
            if(isOddPhase) {
                // rank[0] recv from rank[1]
                if(rank == 0) {
                    float recvData;
                    timeTmp = MPI_Wtime();
                    MPI_Recv(&recvData, 1, MPI_FLOAT, 1, 0, comm, MPI_STATUS_IGNORE);
                    if(recvData < myDataBuf[mySize - 1]) {
                        isSorted = false;
                    }
                    timeComm += MPI_Wtime() - timeTmp;
                    goto LABEL_BARRIER;
                }
                // rank[1] send to rank[0]
                if(rank == 1) {
                    timeTmp = MPI_Wtime();
                    MPI_Send(myDataBuf, 1, MPI_FLOAT, 0, 0, comm);
                    timeComm += MPI_Wtime() - timeTmp;
                }
                if(size % 2 == 0 && size > 2) {
                    // rank[size - 1] recv from rank[size - 2]
                    if(rank == size - 1) {
                        float recvData;
                        timeTmp = MPI_Wtime();
                        MPI_Recv(&recvData, 1, MPI_FLOAT, size - 2, 0, comm, MPI_STATUS_IGNORE);
                        if(recvData > myDataBuf[0]) {
                            isSorted = false;
                        }
                        timeComm += MPI_Wtime() - timeTmp;
                        goto LABEL_BARRIER;
                    }
                    // rank[size - 2] send to rank[size - 1]
                    if(rank == size - 2) {
                        timeTmp = MPI_Wtime();
                        MPI_Send(myDataBuf + mySize - 1, 1, MPI_FLOAT, size - 1, 0, comm);
                        timeComm += MPI_Wtime() - timeTmp;
                    }
                }
            } else if(size % 2 == 1) {
                // rank[size - 1] recv from rank[size - 2]
                if(rank == size - 1) {
                    float recvData;
                    timeTmp = MPI_Wtime();
                    MPI_Recv(&recvData, 1, MPI_FLOAT, size - 2, 0, comm, MPI_STATUS_IGNORE);
                    if(recvData > myDataBuf[0]) {
                        isSorted = false;
                    }
                    timeComm += MPI_Wtime() - timeTmp;
                    goto LABEL_BARRIER;
                }
                // rank[size - 2] send to rank[size - 1]
                if(rank == size - 2) {
                    timeTmp = MPI_Wtime();
                    MPI_Send(myDataBuf + mySize - 1, 1, MPI_FLOAT, size - 1, 0, comm);
                    timeComm += MPI_Wtime() - timeTmp;
                }
            }

            // Normal cases
            if((isOddPhase && rank % 2 == 0) || (!isOddPhase && rank % 2 == 1)) {
                // Send and recv median
                float recvMax;
                timeTmp = MPI_Wtime();
                MPI_Sendrecv(
                    myDataBuf, 1, MPI_FLOAT, rank - 1, 0,
                    &recvMax, 1, MPI_FLOAT, rank - 1, 0,
                    comm, MPI_STATUS_IGNORE
                );
                timeComm += MPI_Wtime() - timeTmp;
                if(recvMax > myDataBuf[0]) {
                    isSorted = false;
                    // Send and recv size
                    int sendSize = getHalfSize(myDataBuf, mySize, recvMax);
                    int recvSize;
                    timeTmp = MPI_Wtime();
                    MPI_Sendrecv(
                        &sendSize, 1, MPI_INT, rank - 1, 0,
                        &recvSize, 1, MPI_INT, rank - 1, 0,
                        comm, MPI_STATUS_IGNORE
                    );
                    // Send and recv data
                    MPI_Sendrecv(
                        myDataBuf, sendSize, MPI_FLOAT, rank - 1, 0,
                        recvDataBuf, recvSize, MPI_FLOAT, rank - 1, 0,
                        comm, MPI_STATUS_IGNORE
                    );
                    timeComm += MPI_Wtime() - timeTmp;
                    // Merge data
                    timeTmp = MPI_Wtime();
                    mergeData(myDataBuf, mySize, recvDataBuf, recvSize, false, funcBuf);
                    timeMerge += MPI_Wtime() - timeTmp;
                }
            } else {
                // Send and recv median
                float recvMin;
                timeTmp = MPI_Wtime();
                MPI_Sendrecv(
                    myDataBuf + mySize - 1, 1, MPI_FLOAT, rank + 1, 0,
                    &recvMin, 1, MPI_FLOAT, rank + 1, 0,
                    comm, MPI_STATUS_IGNORE
                );
                timeComm += MPI_Wtime() - timeTmp;
                if(recvMin < myDataBuf[mySize - 1]) {
                    isSorted = false;
                    // Send and recv size
                    int sendSize = mySize - getHalfSize(myDataBuf, mySize, recvMin);
                    int recvSize;
                    timeTmp = MPI_Wtime();
                    MPI_Sendrecv(
                        &sendSize, 1, MPI_INT, rank + 1, 0,
                        &recvSize, 1, MPI_INT, rank + 1, 0,
                        comm, MPI_STATUS_IGNORE
                    );
                    // Send and recv data
                    MPI_Sendrecv(
                        myDataBuf + mySize - sendSize, sendSize, MPI_FLOAT, rank + 1, 0,
                        recvDataBuf, recvSize, MPI_FLOAT, rank + 1, 0,
                        comm, MPI_STATUS_IGNORE
                    );
                    timeComm += MPI_Wtime() - timeTmp;
                    // Merge data
                    timeTmp = MPI_Wtime();
                    mergeData(myDataBuf, mySize, recvDataBuf, recvSize, true, funcBuf);
                    timeMerge += MPI_Wtime() - timeTmp;
                }
            }

            LABEL_BARRIER:
            isOddPhase = !isOddPhase;
            timeTmp = MPI_Wtime();
            MPI_Allreduce(&isSorted, &isSortedAll, 1, MPI_C_BOOL, MPI_LAND, comm);
            timeComm += MPI_Wtime() - timeTmp;
        }
    }

    // MPI write file
    timeTmp = MPI_Wtime();
    MPI_File fout;
    MPI_File_open(comm, outFileName, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fout);
    MPI_File_write_at_all(fout, offset, myDataBuf, mySize, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&fout);
    timeIO += MPI_Wtime() - timeTmp;
    
    // MPI finalize
    if(rank == 0) {
        double timeTotal = MPI_Wtime() - timeStart;
        std::cout << std::endl;
        std::cout << "Total: " << timeTotal << std::endl;
        std::cout << "------------------" << std::endl;
        std::cout << "Comm: " << timeComm << std::endl;
        std::cout << "IO: " << timeIO << std::endl;
        std::cout << "Merge: " << timeMerge << std::endl;
        std::cout << "Sort: " << timeSort << std::endl;
        std::cout << "Others: " << timeTotal - timeComm - timeIO - timeMerge - timeSort << std::endl;
        std::cout << std::endl;
    }
    MPI_Finalize();
}
