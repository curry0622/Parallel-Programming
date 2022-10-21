#include <mpi.h>
#include <boost/sort/spreadsort/spreadsort.hpp>

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

int main(int argc, char** argv) {
    // MPI init
    MPI_Init(&argc, &argv);
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
    MPI_File fin;
    MPI_File_open(comm, *(argv + 2), MPI_MODE_RDONLY, MPI_INFO_NULL, &fin);
    MPI_File_read_at(fin, offset, myDataBuf, mySize, MPI_FLOAT, MPI_STATUS_IGNORE);

    // Initial sort
    boost::sort::spreadsort::float_sort(myDataBuf, myDataBuf + mySize);

    // Odd even sort
    if(size > 1) {
        bool isSortedAll = false;
        bool isOddPhase = true;
        float recvTarget;
        int baseSendSize = int(base >> 1) + 1;
        int sendSize, recvSize;

        while(!isSortedAll) {
            bool isSorted = true;
            sendSize = baseSendSize;

            // Edge cases
            if(isOddPhase) {
                // rank[0] recv from rank[1]
                if(rank == 0) {
                    float recvData;
                    MPI_Recv(&recvData, 1, MPI_FLOAT, 1, 0, comm, MPI_STATUS_IGNORE);
                    if(recvData < *(myDataBuf + mySize - 1)) {
                        isSorted = false;
                    }
                    goto LABEL_BARRIER;
                }
                // rank[1] send to rank[0]
                if(rank == 1) {
                    MPI_Send(myDataBuf, 1, MPI_FLOAT, 0, 0, comm);
                    if(size == 2) {
                        goto LABEL_BARRIER;
                    }
                }
                if(!(size & 1) && size > 2) {
                    // rank[size - 1] recv from rank[size - 2]
                    if(rank == size - 1) {
                        float recvData;
                        MPI_Recv(&recvData, 1, MPI_FLOAT, size - 2, 0, comm, MPI_STATUS_IGNORE);
                        if(recvData > *myDataBuf) {
                            isSorted = false;
                        }
                        goto LABEL_BARRIER;
                    }
                    // rank[size - 2] send to rank[size - 1]
                    if(rank == size - 2) {
                        MPI_Send(myDataBuf + mySize - 1, 1, MPI_FLOAT, size - 1, 0, comm);
                    }
                }
            } else if(size & 1) {
                // rank[size - 1] recv from rank[size - 2]
                if(rank == size - 1) {
                    float recvData;
                    MPI_Recv(&recvData, 1, MPI_FLOAT, size - 2, 0, comm, MPI_STATUS_IGNORE);
                    if(recvData > *myDataBuf) {
                        isSorted = false;
                    }
                    goto LABEL_BARRIER;
                }
                // rank[size - 2] send to rank[size - 1]
                if(rank == size - 2) {
                    MPI_Send(myDataBuf + mySize - 1, 1, MPI_FLOAT, size - 1, 0, comm);
                }
            }

            // Normal cases
            if((isOddPhase && !(rank & 1)) || (!isOddPhase && (rank & 1))) {
                // Send and recv target
                MPI_Sendrecv(
                    myDataBuf, 1, MPI_FLOAT, rank - 1, 0,
                    &recvTarget, 1, MPI_FLOAT, rank - 1, 0,
                    comm, MPI_STATUS_IGNORE
                );
                if(recvTarget > *(myDataBuf)) {
                    isSorted = false;
                    // Send and recv data
                    sendSize = std::min(sendSize, getFirstBiggerThanTargetIndex(myDataBuf, &mySize, &recvTarget));
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
                    // Merge data
                    mergeData(myDataBuf, &mySize, recvDataBuf, &recvSize, funcBuf, false);
                }
            } else {
                // Send and recv target
                MPI_Sendrecv(
                    myDataBuf + mySize - 1, 1, MPI_FLOAT, rank + 1, 0,
                    &recvTarget, 1, MPI_FLOAT, rank + 1, 0,
                    comm, MPI_STATUS_IGNORE
                );
                if(recvTarget < *(myDataBuf + mySize - 1)) {
                    isSorted = false;
                    // Send and recv data
                    sendSize = std::min(sendSize, mySize - getFirstBiggerThanTargetIndex(myDataBuf, &mySize, &recvTarget));
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
                    // Merge data
                    mergeData(myDataBuf, &mySize, recvDataBuf, &recvSize, funcBuf, true);
                }
            }

            LABEL_BARRIER:
            isOddPhase = !isOddPhase;
            MPI_Allreduce(&isSorted, &isSortedAll, 1, MPI_C_BOOL, MPI_LAND, comm);
        }
    }

    // MPI write file
    MPI_File fout;
    MPI_File_open(comm, *(argv + 3), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fout);
    MPI_File_write_at(fout, offset, myDataBuf, mySize, MPI_FLOAT, MPI_STATUS_IGNORE);

    // MPI finalize
    MPI_Finalize();
}
