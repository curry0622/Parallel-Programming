#include <mpi.h>
#include <boost/sort/spreadsort/float_sort.hpp>
// #include <stdio.h>
// #include <stdlib.h>
// #include <cstring>
// #include <cstdio>
// #include <cfloat>
// #include <algorithm>
// #include <iostream>

int getOffset(const int* rank, const int* size, const int* arrSize) {
    int offset = 0;
    int left = (*arrSize) % (*size);
    int base = (*arrSize) / (*size);
    if (*rank < left)
        offset = *rank * (base + 1);
    else
        offset = left * (base + 1) + (*rank - left) * base;
    return offset;
}

int getFirstBiggerThanTargetIndex(const float* arr, const int* size, const float* target) {
    int low = 0, high = *size;
    while(low != high) {
        int mid = (low + high) / 2;
        if(*(arr + mid) <= *target)
            low = mid + 1;
        else
            high = mid;
    }
    return low;
}

void mergeData(float* myDataBuf, const int* mySize, const float* recvDataBuf, const int* recvSize, float* funcBuf, const bool fromMin) {
    if(fromMin) {
        int i = 0, j = 0;
        while(i < *mySize && j < *recvSize && i + j < *mySize) {
            if(*(myDataBuf + i) < *(recvDataBuf + j)) {
                *(funcBuf + i + j) = *(myDataBuf + i);
                i++;
            } else {
                *(funcBuf + i + j) = *(recvDataBuf + j);
                j++;
            }
        }
        while (i < *mySize && i + j < *mySize) {
            *(funcBuf + i + j) = *(myDataBuf + i);
            i++;
        }
        while (j < *recvSize && i + j < *mySize) {
            *(funcBuf + i + j) = *(recvDataBuf + j);
            j++;
        }
    } else {
        int i = *mySize - 1, j = *recvSize - 1;
        int idx = *mySize - 1;
        while(i >= 0 && j >= 0 && idx >= 0) {
            if(*(myDataBuf + i) > *(recvDataBuf + j)) {
                *(funcBuf + idx) = *(myDataBuf + i);
                i--;
            } else {
                *(funcBuf + idx) = *(recvDataBuf + j);
                j--;
            }
            idx--;
        }
        while(i >= 0 && idx >= 0) {
            *(funcBuf + idx) = *(myDataBuf + i);
            i--;
            idx--;
        }
        while(j >= 0 && idx >= 0) {
            *(funcBuf + idx) = *(recvDataBuf + j);
            j--;
            idx--;
        }
    }
    memcpy(myDataBuf, funcBuf, sizeof(float) * (*mySize));
}

int main(int argc, char** argv) {
    // Time measurement
    // double timeIO = 0;
    // double timeIORead = 0;
    // double timeIOWrite = 0;
    // double timeComm = 0;
    // double timeCommBuf = 0;
    // double timeCPU = 0;
    // double timeCPUSort = 0;
    // double timeCPUMerge = 0;
    // double timeTmp = 0;
    // double timeTmp2 = 0;
    // double timeLoop = 0;

    // MPI init
    MPI_Init(&argc, &argv);
    // double timeStart = MPI_Wtime();
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get args
    const int arrSize = atoi(*(argv + 1));

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
    MPI_Offset offset = getOffset(&rank, &size, &arrSize) * sizeof(float);

    // Allocate memory
    float* myDataBuf = (float*)malloc(sizeof(float) * (mySize + 1));
    float* recvDataBuf = (float*)malloc(sizeof(float) * (mySize + 1));
    float* funcBuf = (float*)malloc(sizeof(float) * (mySize + 1));

    // MPI read file
    // timeTmp = MPI_Wtime();
    MPI_File fin;
    MPI_File_open(comm, *(argv + 2), MPI_MODE_RDONLY, MPI_INFO_NULL, &fin);
    MPI_File_read_at(fin, offset, myDataBuf, mySize, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&fin);
    // timeIO += MPI_Wtime() - timeTmp;
    // timeIORead = timeIO;

    // Initial sort
    // timeTmp = MPI_Wtime();
    boost::sort::spreadsort::float_sort(myDataBuf, myDataBuf + mySize);
    // RadixSort8(myDataBuf, mySize);
    // timeCPUSort += MPI_Wtime() - timeTmp;

    // Odd even sort
    if(size > 1) {
        bool isSortedAll = false;
        bool isOddPhase = true;

        // timeTmp2 = MPI_Wtime();
        while(!isSortedAll) {
            bool isSorted = true;

            // Edge cases
            if(isOddPhase) {
                // rank[0] recv from rank[1]
                if(rank == 0) {
                    float recvData;
                    // timeTmp = MPI_Wtime();
                    MPI_Recv(&recvData, 1, MPI_FLOAT, 1, 0, comm, MPI_STATUS_IGNORE);
                    if(recvData < *(myDataBuf + mySize - 1)) {
                        isSorted = false;
                    }
                    // timeComm += MPI_Wtime() - timeTmp;
                    goto LABEL_BARRIER;
                }
                // rank[1] send to rank[0]
                if(rank == 1) {
                    // timeTmp = MPI_Wtime();
                    MPI_Send(myDataBuf, 1, MPI_FLOAT, 0, 0, comm);
                    // timeComm += MPI_Wtime() - timeTmp;
                }
                if(!(size & 1) && size > 2) {
                    // rank[size - 1] recv from rank[size - 2]
                    if(rank == size - 1) {
                        float recvData;
                        // timeTmp = MPI_Wtime();
                        MPI_Recv(&recvData, 1, MPI_FLOAT, size - 2, 0, comm, MPI_STATUS_IGNORE);
                        if(recvData > *myDataBuf) {
                            isSorted = false;
                        }
                        // timeComm += MPI_Wtime() - timeTmp;
                        goto LABEL_BARRIER;
                    }
                    // rank[size - 2] send to rank[size - 1]
                    if(rank == size - 2) {
                        // timeTmp = MPI_Wtime();
                        MPI_Send(myDataBuf + mySize - 1, 1, MPI_FLOAT, size - 1, 0, comm);
                        // timeComm += MPI_Wtime() - timeTmp;
                    }
                }
            } else if(size & 1) {
                // rank[size - 1] recv from rank[size - 2]
                if(rank == size - 1) {
                    float recvData;
                    // timeTmp = MPI_Wtime();
                    MPI_Recv(&recvData, 1, MPI_FLOAT, size - 2, 0, comm, MPI_STATUS_IGNORE);
                    if(recvData > *myDataBuf) {
                        isSorted = false;
                    }
                    // timeComm += MPI_Wtime() - timeTmp;
                    goto LABEL_BARRIER;
                }
                // rank[size - 2] send to rank[size - 1]
                if(rank == size - 2) {
                    // timeTmp = MPI_Wtime();
                    MPI_Send(myDataBuf + mySize - 1, 1, MPI_FLOAT, size - 1, 0, comm);
                    // timeComm += MPI_Wtime() - timeTmp;
                }
            }

            // Normal cases
            if((isOddPhase && !(rank & 1)) || (!isOddPhase && (rank & 1))) {
                // Send and recv target
                float recvMax;
                // timeTmp = MPI_Wtime();
                MPI_Sendrecv(
                    myDataBuf, 1, MPI_FLOAT, rank - 1, 0,
                    &recvMax, 1, MPI_FLOAT, rank - 1, 0,
                    comm, MPI_STATUS_IGNORE
                );
                // timeComm += MPI_Wtime() - timeTmp;
                if(recvMax > *(myDataBuf)) {
                    isSorted = false;
                    // Send and recv size
                    int sendSize = getFirstBiggerThanTargetIndex(myDataBuf, &mySize, &recvMax);
                    int recvSize;
                    // timeTmp = MPI_Wtime();
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
                    // timeCommBuf += MPI_Wtime() - timeTmp;
                    // timeComm += MPI_Wtime() - timeTmp;
                    // Merge data
                    // timeTmp = MPI_Wtime();
                    mergeData(myDataBuf, &mySize, recvDataBuf, &recvSize, funcBuf, false);
                    // timeCPUMerge += MPI_Wtime() - timeTmp;
                }
            } else {
                // Send and recv target
                float recvMin;
                // timeTmp = MPI_Wtime();
                MPI_Sendrecv(
                    myDataBuf + mySize - 1, 1, MPI_FLOAT, rank + 1, 0,
                    &recvMin, 1, MPI_FLOAT, rank + 1, 0,
                    comm, MPI_STATUS_IGNORE
                );
                // timeComm += MPI_Wtime() - timeTmp;
                if(recvMin < *(myDataBuf + mySize - 1)) {
                    isSorted = false;
                    // Send and recv size
                    int sendSize = mySize - getFirstBiggerThanTargetIndex(myDataBuf, &mySize, &recvMin);
                    int recvSize;
                    // timeTmp = MPI_Wtime();
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
                    // timeCommBuf += MPI_Wtime() - timeTmp;
                    // timeComm += MPI_Wtime() - timeTmp;
                    // Merge data
                    // timeTmp = MPI_Wtime();
                    mergeData(myDataBuf, &mySize, recvDataBuf, &recvSize, funcBuf, true);
                    // timeCPUMerge += MPI_Wtime() - timeTmp;
                }
            }

            LABEL_BARRIER:
            isOddPhase = !isOddPhase;
            MPI_Allreduce(&isSorted, &isSortedAll, 1, MPI_C_BOOL, MPI_LAND, comm);
        }
        // timeLoop += MPI_Wtime() - timeTmp2;
    }

    // MPI write file
    // timeTmp = MPI_Wtime();
    MPI_File fout;
    MPI_File_open(comm, *(argv + 3), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fout);
    MPI_File_write_at(fout, offset, myDataBuf, mySize, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&fout);
    // timeIO += MPI_Wtime() - timeTmp;
    // timeIOWrite = timeIO - timeIORead;
    
    // MPI finalize
    // if(rank == size / 2) {
    //     double timeTotal = MPI_Wtime() - timeStart, timeCPU = timeTotal - timeComm - timeIO;
    //     std::cout << std::endl;
    //     std::cout << "Total: " << timeTotal << std::endl;
    //     std::cout << "------------------" << std::endl;
    //     std::cout << "[Comm]: " << timeComm << " -> buf: " << timeCommBuf << "(" << timeCommBuf / timeComm << ")\n";
    //     std::cout << "[IO]: " << timeIO << " -> ";
    //     std::cout << "Read: " << timeIORead << "(" << timeIORead / timeIO << "), ";
    //     std::cout << "Write: " << timeIOWrite << "(" << timeIOWrite / timeIO << ")" << std::endl;
    //     std::cout << "[CPU]: " << timeCPU << " -> ";
    //     std::cout << "Merge: " << timeCPUMerge << "(" << timeCPUMerge / timeCPU << "), ";
    //     std::cout << "Sort: " << timeCPUSort << "(" << timeCPUSort / timeCPU << "), ";
    //     std::cout << "Else: " << timeCPU - timeCPUMerge - timeCPUSort << "(" << (timeCPU - timeCPUMerge - timeCPUSort) / timeCPU << ")" << std::endl;
    //     std::cout << "[Loop]: " << timeLoop << std::endl;
    //     std::cout << "------------------" << std::endl;
    //     std::cout << std::endl;
    // }
    MPI_Finalize();
}
