#ifndef COMMANDLINE_H
#define COMMANDLINE_H

#include <fstream>
#include <iostream>
#include <cstdlib>
#include <string>
#include "Pb_Data.h"
using namespace std;

class commandline
{
private:
	// is the commandline valid ?
	bool command_ok;

	// path of the dataset
	string instance_path;

    // name of the instance file
    string instance_filename;

	// output path
	string output_path;

	// type of problem
	ProblemType problemType;

	// string describing type of problem
	string strProblemType;

	// back-ordering cost
	double backorder_cost;

	// holding cost
	double holding_cost;

	// proportion of samples in the training set relative to the total
	double train_val_split;

	// size of the subset used for each split in the BL Shuffle & Split
	int split_size;

	// proportion of features to be used in each split in the BL Shuffle & Split
	double split_feat;

	// regularization parameter for ERM models (lambda)
	double regularization_param;

	// number of folds for cross validation
	int nb_folds;

	// number of breakpoints for grid search
	int nb_breakpoints;

	// informative factor
	double informativeFactor;

	// set value for variable z
	bool set_value_z;

	// number of threads for CPLEX
	int nbThreads;

	// time limit for CPLEX
	double timeLimit;

	// set the solver type
	int set_problem_type(string to_parse);

    // display the name of the problem
	void display_problem_name(string to_parse);

public:
	// constructor
	commandline(int argc, char *argv[]);

	// destructor
	~commandline();

	// gets the path to the instance
	string get_instance_path();

    // gets the instance filename
    string get_instance_filename();

	// gets output path
	string get_output_path();

	// gets back-ordering cost
	double get_backorder_cost();

	// gets holding cost
	double get_holding_cost();

	// gets the solver type
	ProblemType get_problem_type();

	// gets string describing problem type
	string get_str_problem_type();

	// gets train-validation split (the proportion of samples from the complete data set to be in the training set)
	double get_train_val_split();

	// gets the size of the subset to be used for splitting the data into training/validation sets
	int get_split_size();

	// gets the proportion of features to be used in each split
	double get_split_feat();

	// gets regularization parameter
	double get_regularization_param();

	// gets number of folds for cross validation
	int get_nb_folds();

	// gets number of breakpoints for grid search
	int get_nb_breakpoints();

	// get number of threads
	int get_nbThreads();

	// get CPLEX time limit
	double get_timeLimit();

	// get informative factor
	double get_informative_factor();

	// set value for variable z ?
	bool get_set_value_z();

	// is the commandline valid ?
	bool is_valid();
};

#endif