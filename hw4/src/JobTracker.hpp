#ifndef JOBTRACKER_HPP
#define JOBTRACKER_HPP

#include <bits/stdc++.h>
#include <mpi.h>

class JobTracker {
public:
    // Variables
    int num_nodes;
    int num_chunks;
    std::map<int, int> loc_config;
    std::string job_name;
    std::string output_dir;

    // Constructor
    JobTracker(int num_nodes, std::string job_name, std::string loc_config_file, std::string output_dir);

    // Methods
    void set_loc_config(std::string loc_config_file);
    void dispatch_map_tasks();
    void shuffle();

    // Utils
    void print_loc_config();
    void verify();
};

#endif