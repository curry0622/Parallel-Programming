#include <bits/stdc++.h>

#ifndef JOBTRACKER_HPP
#define JOBTRACKER_HPP

class JobTracker {
public:
    // Variables
    int num_nodes;
    std::map<int, int> loc_config;

    // Constructor
    JobTracker(int num_nodes, std::string loc_config_file);

    // Methods
    void set_loc_config(std::string loc_config_file);

    // Utils
    void print_loc_config();
};

#endif