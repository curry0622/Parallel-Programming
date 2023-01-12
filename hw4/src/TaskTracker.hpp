#ifndef TASKTRACKER_HPP
#define TASKTRACKER_HPP

#include <bits/stdc++.h>
#include <mpi.h>
#include <pthread.h>

class TaskTracker {
public:
    /* Variables */
    int num_cpus;
    int node_id;
    int chunk_size;
    int delay;
    int num_reducers;
    std::string word_file;
    static pthread_mutex_t mutex;
    static pthread_cond_t cond;
    static std::queue<std::pair<int, int>> tasks;

    /* Constructor */
    TaskTracker(int node_id, int chunk_size, int delay, int num_reducers, std::string word_file);

    /* Methods */
    std::pair<int, int> get_task();
    void set_num_cpus();
    void req_map_tasks();
    static void* map_thread_func(void* thread_id);
    std::map<int, std::string> input_split(int chunk_id);
    std::map<std::string, int> map(std::pair<int, std::string> record);
    int partition(std::string key);

    /* Utils */
    void print();
};

#endif