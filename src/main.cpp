#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <stdlib.h>
#include <bits/stdc++.h> 
#include <sys/stat.h> 
#include <sys/types.h> 
#include <chrono>
#include <ctime> 
#include <exception>

#include "commandline.h"
#include "Solution.h"
#include "Solver.h"

int main(int argc, char *argv[])
{
    try
    {
        double solution_time;
        
        commandline c(argc, argv);
        if (c.is_valid())
        {
            // Parsing the problem instance
            auto myData = std::make_shared<Pb_Data>(c.get_instance_path(), 
                             c.get_instance_filename(), c.get_output_path(),
                             c.get_backorder_cost(), c.get_holding_cost(), 
                             c.get_problem_type(), c.get_str_problem_type(), 
                             c.get_train_val_split(), c.get_split_size(), 
                             c.get_split_feat(), c.get_regularization_param(), 
                             c.get_nb_folds(), c.get_nb_breakpoints(),
                             c.get_nbThreads(), c.get_timeLimit(), 
                             c.get_set_value_z());
            
            // Begin of clock
            myData->time_StartOpt = clock();
            auto wallClockTimeStart = std::chrono::system_clock::now();

            // Create the solution data structure
            auto mySolution = Solution::createSolution(myData);

            // Create the solver data structures
            auto mySolver = Solver::createSolver(myData, mySolution);

            // Solve the problem
            if (mySolver->solve())
                return 1;

            // End of clock
            myData->time_EndOpt = clock();
            solution_time = (((double)(myData->time_EndOpt - myData->time_StartOpt)) / CLOCKS_PER_SEC);
            mySolution->solution_time = solution_time;
            std::chrono::duration<double> wallClockTimeDuration = (std::chrono::system_clock::now() - wallClockTimeStart);
            std::cout << "END OF ALGORITHM" << std::endl;
            std::cout << "CPU TIME:        " << solution_time << std::endl;
            std::cout << "WALL CLOCK TIME: " << wallClockTimeDuration.count() << std::endl;
            std::cout << "CPU UTILIZATION: " << solution_time / wallClockTimeDuration.count() << std::endl;

            mySolution->displaySolution();
            mySolution->exportSolution(".out");
        }
        else
            return 1;

        std::cout << "End of execution." << endl;
        return 0; 
    }
    catch ( const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 2;
    }
}