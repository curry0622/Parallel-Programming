#ifndef TASKTRACKER_HPP
#define TASKTRACKER_HPP

#include <bits/stdc++.h>
#include <mpi.h>
#include <pthread.h>
#include <unistd.h>

class TaskTracker {
public:
    /* Variables */
    int num_cpus;
    int node_id;
    static int chunk_size;
    int num_reducers;
    static std::string word_file;
    static int delay;
    static pthread_mutex_t mutex;
    static pthread_cond_t cond;
    static std::queue<std::pair<int, int>> tasks;
    static int num_working;
    static pthread_mutex_t mutex2;
    static pthread_cond_t cond2;

    /* Constructor */
    TaskTracker(int node_id, int chunk_size, int delay, int num_reducers, std::string word_file);

    /* Methods */
    std::pair<int, int> get_task();
    void set_num_cpus();
    void req_map_tasks();
    static void* map_thread_func(void* thread_id);
    static std::map<int, std::string> input_split(int chunk_id);
    static std::map<std::string, int> map(std::pair<int, std::string> record);
    int partition(std::string key);

    /* Utils */
    void print();
};

#endif