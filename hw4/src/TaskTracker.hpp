#include <bits/stdc++.h>
#include <mpi.h>

#ifndef TASKTRACKER_HPP
#define TASKTRACKER_HPP

class TaskTracker {
public:
    // Variables
    int num_cpus;
    int node_id;
    int chunk_size;
    int delay;
    int num_reducers;
    std::string word_file;

    // Constructor
    TaskTracker(int node_id, int chunk_size, int delay, int num_reducers, std::string word_file);

    // Methods
    void set_num_cpus();
    void req_map_tasks();
    int get_chunk_id();
    static void* map_thread_func(void* thread_id);
    std::map<int, std::string> input_split(int chunk_id);
    std::map<std::string, int> map(std::pair<int, std::string> record);
    int partition(std::string key);

    // Utils
    void print();
};

#endif