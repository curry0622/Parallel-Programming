#include <mpi.h>
#include <boost/sort/spreadsort/spreadsort.hpp>
// #include <stdio.h>
// #include <stdlib.h>
// #include <cstring>
// #include <cstdio>
// #include <cfloat>
// #include <algorithm>
#include <iostream>

// int cmp(const void * a, const void * b) {
//    float fa = *(float *)a;
//    float fb = *(float *)b;
//    return (fa > fb) ? 1 : -1;
// }

int getFirstBiggerThanTargetIndex(const float* arr, const int* size, const float* target) {
    int low = 0, high = *size;
    while(low != high) {
        int mid = (low + high) >> 1;
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

void writeResult(float ratio, int test, int iter, long size, double timeCPU, double timeComm) {
    FILE* fp = fopen("ratio.csv", "a");
    fprintf(fp, "%f,%d,%d,%ld,%lf,%lf,%lf\n", ratio, test, iter, size, timeCPU, timeComm, timeCPU + timeComm);
}

int main(int argc, char** argv) {
    // Time measurement
    double timeIO = 0;
    double timeComm = 0;
    double timeCPU = 0;
    double timeTmp = 0;
    double timeTmp2 = 0;
    double timeNormal = 0;

    // MPI init
    MPI_Init(&argc, &argv);
    double timeStart = MPI_Wtime();
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

    // Set mySize and calc offset
    MPI_Offset offset = sizeof(float);
    int base = arrSize / size;
    int left = arrSize % size;
    int mySize = base;
    if (rank < left) {
        offset *= (rank * (base + 1));
        mySize++;
    } else {
        offset *= (left * (base + 1) + (rank - left) * base);
    }

    // Allocate memory
    float* myDataBuf = (float*)malloc(sizeof(float) * (mySize + 1));
    float* recvDataBuf = (float*)malloc(sizeof(float) * (mySize + 1));
    float* funcBuf = (float*)malloc(sizeof(float) * (mySize + 1));

    // MPI read file
    timeTmp = MPI_Wtime();
    MPI_File fin;
    MPI_File_open(comm, *(argv + 2), MPI_MODE_RDONLY, MPI_INFO_NULL, &fin);
    MPI_File_read_at(fin, offset, myDataBuf, mySize, MPI_FLOAT, MPI_STATUS_IGNORE);
    // MPI_File_close(&fin);
    timeIO += MPI_Wtime() - timeTmp;

    // Initial sort
    boost::sort::spreadsort::float_sort(myDataBuf, myDataBuf + mySize);

    int iteration = 0;
    long totalSendSize = 0;
    // Odd even sort
    if(size > 1) {
        bool isSortedAll = false;
        bool isSorted = true;
        bool isOddPhase = true;
        float recvTarget;
        int sendSize, recvSize;
        // int orgSendSize = int(base * atof(*(argv + 4))) + 1;
        int orgSendSize = int(base * 0.5) + 1;
        // int orgSendSize = mySize;
        sendSize = orgSendSize;
        recvSize = sendSize;

        while(!isSortedAll) {
            isSorted = true;
            iteration++;
            sendSize = orgSendSize;

            // Edge cases
            if(isOddPhase) {
                // rank[0] recv from rank[1]
                if(rank == 0) {
                    float recvData;
                    timeTmp = MPI_Wtime();
                    MPI_Recv(&recvData, 1, MPI_FLOAT, 1, 0, comm, MPI_STATUS_IGNORE);
                    if(recvData < *(myDataBuf + mySize - 1)) {
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
                    if(size == 2) {
                        goto LABEL_BARRIER;
                    }
                }
                if(!(size & 1) && size > 2) {
                    // rank[size - 1] recv from rank[size - 2]
                    if(rank == size - 1) {
                        float recvData;
                        timeTmp = MPI_Wtime();
                        MPI_Recv(&recvData, 1, MPI_FLOAT, size - 2, 0, comm, MPI_STATUS_IGNORE);
                        if(recvData > *myDataBuf) {
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
            } else if(size & 1) {
                // rank[size - 1] recv from rank[size - 2]
                if(rank == size - 1) {
                    float recvData;
                    timeTmp = MPI_Wtime();
                    MPI_Recv(&recvData, 1, MPI_FLOAT, size - 2, 0, comm, MPI_STATUS_IGNORE);
                    if(recvData > *myDataBuf) {
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

            timeTmp2 = MPI_Wtime();
            // Normal cases
            if((isOddPhase && !(rank & 1)) || (!isOddPhase && (rank & 1))) {
                // Send and recv target
                timeTmp = MPI_Wtime();
                MPI_Sendrecv(
                    myDataBuf, 1, MPI_FLOAT, rank - 1, 0,
                    &recvTarget, 1, MPI_FLOAT, rank - 1, 0,
                    comm, MPI_STATUS_IGNORE
                );
                timeComm += MPI_Wtime() - timeTmp;
                if(recvTarget > *(myDataBuf)) {
                    isSorted = false;
                    // Send and recv data
                    sendSize = std::min(sendSize, getFirstBiggerThanTargetIndex(myDataBuf, &mySize, &recvTarget));
                    // sendSize = getFirstBiggerThanTargetIndex(myDataBuf, &mySize, &recvTarget);
                    totalSendSize += sendSize;
                    timeTmp = MPI_Wtime();
                    MPI_Sendrecv(
                        &sendSize, 1, MPI_INT, rank - 1, 0,
                        &recvSize, 1, MPI_INT, rank - 1, 0,
                        comm, MPI_STATUS_IGNORE
                    );
                    MPI_Sendrecv(
                        myDataBuf, sendSize, MPI_FLOAT, rank - 1, 0,
                        recvDataBuf, recvSize, MPI_FLOAT, rank - 1, 0,
                        comm, MPI_STATUS_IGNORE
                    );
                    timeComm += MPI_Wtime() - timeTmp;
                    // Merge data
                    mergeData(myDataBuf, &mySize, recvDataBuf, &recvSize, funcBuf, false);
                }
            } else {
                // Send and recv target
                timeTmp = MPI_Wtime();
                MPI_Sendrecv(
                    myDataBuf + mySize - 1, 1, MPI_FLOAT, rank + 1, 0,
                    &recvTarget, 1, MPI_FLOAT, rank + 1, 0,
                    comm, MPI_STATUS_IGNORE
                );
                timeComm += MPI_Wtime() - timeTmp;
                if(recvTarget < *(myDataBuf + mySize - 1)) {
                    isSorted = false;
                    // Send and recv data
                    sendSize = std::min(sendSize, mySize - getFirstBiggerThanTargetIndex(myDataBuf, &mySize, &recvTarget));
                    // sendSize = mySize - getFirstBiggerThanTargetIndex(myDataBuf, &mySize, &recvTarget);
                    totalSendSize += sendSize;
                    timeTmp = MPI_Wtime();
                    MPI_Sendrecv(
                        &sendSize, 1, MPI_INT, rank + 1, 0,
                        &recvSize, 1, MPI_INT, rank + 1, 0,
                        comm, MPI_STATUS_IGNORE
                    );
                    MPI_Sendrecv(
                        myDataBuf + mySize - sendSize, sendSize, MPI_FLOAT, rank + 1, 0,
                        recvDataBuf, recvSize, MPI_FLOAT, rank + 1, 0,
                        comm, MPI_STATUS_IGNORE
                    );
                    timeComm += MPI_Wtime() - timeTmp;
                    // Merge data
                    mergeData(myDataBuf, &mySize, recvDataBuf, &recvSize, funcBuf, true);
                }
            }
            timeNormal += MPI_Wtime() - timeTmp2;

            LABEL_BARRIER:
            isOddPhase = !isOddPhase;
            MPI_Allreduce(&isSorted, &isSortedAll, 1, MPI_C_BOOL, MPI_LAND, comm);
        }
    }

    // MPI write file
    timeTmp = MPI_Wtime();
    MPI_File fout;
    MPI_File_open(comm, *(argv + 3), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fout);
    MPI_File_write_at(fout, offset, myDataBuf, mySize, MPI_FLOAT, MPI_STATUS_IGNORE);
    // MPI_File_close(&fout);
    timeIO += MPI_Wtime() - timeTmp;
    
    // MPI finalize
    if(rank == size / 2) {
        double timeTotal = MPI_Wtime() - timeStart;
        double timeCPU = timeTotal - timeComm - timeIO;
        std::cout << std::endl;
        std::cout << "Total: " << timeTotal << std::endl;
        std::cout << "------------------" << std::endl;
        std::cout << "[Comm]: " << timeComm << std::endl;
        std::cout << "[IO]: " << timeIO << std::endl;
        std::cout << "[CPU]: " << timeCPU << std::endl;
        std::cout << "------------------" << std::endl;
        std::cout << std::endl;
        // writeResult(timeComm, timeIO, timeCPU, timeTotal);
    }

    
    // for(int i = 0; i < size; i++) {
    //     if(rank == i) {
    //         std::cout << "Rank " << rank << ", ";
    //         std::cout << "iteration: " << iteration << ", ";
    //         std::cout << "total: " << timeTotal << ", ";
    //         std::cout << "comm: " << timeComm << ", ";
    //         std::cout << "io: " << timeIO << ", ";
    //         std::cout << "cpu: " << timeCPU << ", ";
    //         std::cout << "send size: " << totalSendSize << std::endl;
    //     }
    //     MPI_Barrier(comm);
    // }

    // double timeTotal = MPI_Wtime() - timeStart;
    // timeCPU = timeTotal - timeComm - timeIO;
    // double avgTimeTotal = 0;
    // double avgTimeComm = 0;
    // double avgTimeIO = 0;
    // double avgTimeCPU = 0;
    // long avgTotalSendSize = 0;
    // MPI_Allreduce(&timeTotal, &avgTimeTotal, 1, MPI_DOUBLE, MPI_SUM, comm);
    // MPI_Allreduce(&timeComm, &avgTimeComm, 1, MPI_DOUBLE, MPI_SUM, comm);
    // MPI_Allreduce(&timeIO, &avgTimeIO, 1, MPI_DOUBLE, MPI_SUM, comm);
    // MPI_Allreduce(&timeCPU, &avgTimeCPU, 1, MPI_DOUBLE, MPI_SUM, comm);
    // MPI_Allreduce(&totalSendSize, &avgTotalSendSize, 1, MPI_LONG, MPI_SUM, comm);
    // if(rank == 0) {
    //     std::cout << std::endl;
    //     std::cout << "------------------" << std::endl;
    //     std::cout << "[Iteration]: " << iteration << std::endl;
    //     std::cout << "[Comm]: " << avgTimeComm / size << std::endl;
    //     std::cout << "[CPU]: " << avgTimeCPU / size << std::endl;
    //     std::cout << "[Comm+CPU]: " << (avgTimeComm + avgTimeCPU) / size << std::endl;
    //     std::cout << "[Send Size]: " << avgTotalSendSize / size << std::endl;
    //     std::cout << "------------------" << std::endl;
    //     std::cout << std::endl;
    //     writeResult(atof(*(argv + 4)), atoi(*(argv + 5)), iteration, avgTotalSendSize / size, avgTimeCPU / size, avgTimeComm / size);
    // }

    MPI_Finalize();

}
