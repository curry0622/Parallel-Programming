#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <iostream>

unsigned long long r = 0;
unsigned long long rSqr = 0;
unsigned long long k = 0;
unsigned long long pixels = 0;
unsigned long long threadNum = 0;

struct threadArgs {
    int threadID;
};

void* calcPixels(void* tArgs) {
    // Get args
    threadArgs* args = (threadArgs*)tArgs;
    int threadID = args->threadID;
    unsigned long long tmpPixels = 0;

    // Calculate range
    unsigned long long start = threadID * (r / threadNum);
    unsigned long long end = (threadID + 1) * (r / threadNum);
    if (threadID == threadNum - 1) {
        end = r;
    }

    // Calculate pixels
    for (unsigned long long x = start; x < end; x++) {
        unsigned long long y = ceil(sqrtl(rSqr - x*x));
        tmpPixels += y;
    }

    // Store pixels
    pthread_mutex_t mutex;
    pthread_mutex_init(&mutex, NULL);
    pthread_mutex_lock(&mutex);
    pixels += tmpPixels;
    pthread_mutex_unlock(&mutex);
    pthread_mutex_destroy(&mutex);

    // Exit
    pthread_exit(NULL);
}

int main(int argc, char** argv) {
    // Dummy check
    if (argc != 3) {
        fprintf(stderr, "must provide exactly 2 arguments!\n");
        return 1;
    }

    // Get args
    r = atoll(argv[1]);
    k = atoll(argv[2]);
    rSqr = r * r;

    // Get CPU nums
    cpu_set_t cpuSet;
    sched_getaffinity(0, sizeof(cpuSet), &cpuSet);
    unsigned long long cpuNum = CPU_COUNT(&cpuSet);

    // Creaet threads
    threadNum = cpuNum * 2;
    pthread_t threads[threadNum];
    int threadID[threadNum];

    // Calculate pixels
    for(int i = 0; i < threadNum; i++) {
        threadID[i] = i;
        struct threadArgs* args = (struct threadArgs*)malloc(sizeof(struct threadArgs));
        args->threadID = threadID[i];
        int rc = pthread_create(&threads[i], NULL, calcPixels, (void*)args);
        if(rc) {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    // Join threads
    for(int i = 0; i < threadNum; i++) {
        pthread_join(threads[i], NULL);
    }

    // Print pixels
    printf("%llu\n", (4 * (pixels % k)) % k);

    // Exit
    pthread_exit(NULL);
}
