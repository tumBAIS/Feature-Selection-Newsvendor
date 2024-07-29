#include "commandline.h"

int commandline::set_problem_type(string to_parse)
{
	if (to_parse == "BL") // Bilevel Feature Selection
		problemType = BL;
	else if (to_parse == "BL_SS")
		problemType = BL_SS;
	else if (to_parse == "ERM")
		problemType = ERM;
	else if (to_parse == "ERM_l0")
		problemType = ERM_l0;
	else if (to_parse == "ERM_l1")
		problemType = ERM_l1;
	else if (to_parse == "SP_ERM_l0")
		problemType = SP_ERM_l0;
	else if (to_parse == "SP_ERM_l1")
		problemType = SP_ERM_l1;
	else if (to_parse == "GS_ERM_l0")
		problemType = GS_ERM_l0;
	else if (to_parse == "GS_ERM_l1")
		problemType = GS_ERM_l1;
	else if (to_parse == "GS_SS_ERM_l0")
		problemType = GS_SS_ERM_l0;
	else if (to_parse == "GS_SS_ERM_l1")
		problemType = GS_SS_ERM_l1;
	else
		return -1; // problem
	return 0; // OK
}

void commandline::display_problem_name(string to_parse)
{
    string problem_name = this->get_instance_filename();
	cout << "DATASET: " << problem_name << endl;
}

// constructor
commandline::commandline(int argc, char *argv[])
{
	this->command_ok = false;

	//Default parameter values
	this->problemType = BL;
	this->nbThreads = 1;
	this->timeLimit = -1; // use CPLEX default
	this->train_val_split = 0.7;
	this->split_size = -1; // use all data points in the splits
	this->split_feat = 1; // use all features in the splits
	this->regularization_param = 0;
	this->nb_folds = -1;
	this->nb_breakpoints = -1;
	this->output_path = "Output/";
	this->informativeFactor = -1; // do not include these constraints
	this->set_value_z = false;

	int numRequiredParams = 5;
	int numDefaultParams = 22;

	// Read in options
    if ((argc % 2 != (numRequiredParams % 2)) || (argc > numRequiredParams+numDefaultParams) || (argc < numRequiredParams))
	{
		cout << "ERROR: invalid command line" << endl;
		cout << "USAGE: ./executable problem_type instance_path backorder_cost holding_cost [-split train_val_split] [-split_size subset_samples] [-split_feat subset_features] [-lambda regularization_param] [-folds k] [-breakpts n_bpts] [-t time_limit] [-threads nb_threads] [-o out_path] [-i informative_factor] [-setz set_val_z]" << endl;
		
		printf("I received the following %d arguments: \n", argc);
		for (int i=0; i<argc; i++)
		{
			printf("i=%d \t argv[%d] = %s \n", i, i, argv[i]);
		}
		return;
	}
	else
	{
		// Problem type
		if (set_problem_type(string(argv[1])) != 0)
		{
			cout << "ERROR: Unrecognized problem type : " + string(argv[1]) << endl;
			return;
		}
		this->strProblemType = string(argv[1]);

		// Training data path
		this->instance_path = string(argv[2]);
		if (this->instance_path.size() == 0)
		{
			cout << "ERROR: path_to_instance must be provided" << endl;
			return;
		}
		else
		{
			ifstream file(this->instance_path.c_str());
			if (!file.is_open())
			{
				cout << "ERROR: file does not exist" << endl;
				return;
			}
			file.close();
		}

		// Back-ordering cost
		this->backorder_cost = atof(argv[3]);
		if (this->backorder_cost < 0)
		{
			cout << "ERROR: backorder_cost must be a non-negative real number" << endl;
			return;
		}

		// Holding cost
		this->holding_cost = atof(argv[4]);
		if (this->holding_cost < 0)
		{
			cout << "ERROR: holding_cost must be a non-negative real number" << endl;
			return;
		}

		display_problem_name(this->instance_path);
		
		for (int i=numRequiredParams; i<argc; i+=2)
		{
			if (string(argv[i]) == "-split") // Train-Validation Split
			{
				this->train_val_split = atof(argv[i+1]);
				if (this->train_val_split < 0 || this->train_val_split > 1)
				{
					cout << "ERROR: train_val_split must be a real number between 0 and 1" << endl;
					return;
				}
			}

			else if (string(argv[i]) == "-split_size")
			{
				this->split_size = atoi(argv[i+1]);
				if (this->split_size <= 0)
				{
					cout << "ERROR: subset_samples must be an integer number greater than 0" << endl;
					return;
				}
			}

			else if (string(argv[i]) == "-split_feat")
			{
				this->split_feat = atof(argv[i+1]);
				if (this->split_feat < 0 || this->split_feat > 1)
				{
					cout << "ERROR: subset_features must be a real number between 0 and 1" << endl;
					return;
				}
			}

			else if (string(argv[i]) == "-lambda")
				this->regularization_param = atof(argv[i+1]);

			else if (string(argv[i]) == "-folds")
			{
				this->nb_folds = atoi(argv[i+1]);
				if (this->nb_folds <= 0)
				{
					cout << "ERROR: nb_folds must be an integer number greater than 0" << endl;
					return;
				}
			}

			else if (string(argv[i]) == "-breakpts")
				this->nb_breakpoints = atoi(argv[i+1]);

			else if (string(argv[i]) == "-t")
				this->timeLimit = atof(argv[i+1]);

			else if (string(argv[i]) == "-threads")
				this->nbThreads = atoi(argv[i+1]);

			else if (string(argv[i]) == "-o")
				this->output_path = argv[i+1];

			else if (string(argv[i]) == "-i")
				this->informativeFactor = atof(argv[i+1]);

			else if (string(argv[i]) == "-setz")
			{
				if (strcmp(argv[i+1], "y") == 0)
					this->set_value_z = true;
				else if (strcmp(argv[i+1], "n") == 0)
					this->set_value_z = false;
				else
				{
					cout << "ERROR: invalid option for parameter -setz" << endl;
					return;
				}
			}

		}
	}

	// Check for illegal combinations of options
	bool exists;

	// Number of folds - not permitted
	ProblemType disallow_nbfolds[6] = {ERM, ERM_l0, ERM_l1, BL, GS_ERM_l0, GS_ERM_l1};
	exists = std::find(std::begin(disallow_nbfolds), std::end(disallow_nbfolds), this->problemType) != std::end(disallow_nbfolds);
	if (exists == true && this->nb_folds >= 0)
	{
		cout << "ERROR: the chosen method does not support number of folds nb_folds." << endl;
		return;
	}
	// Number of folds - required
	ProblemType require_nbfolds[3] = {BL_SS, GS_SS_ERM_l0, GS_SS_ERM_l1};
	exists = std::find(std::begin(require_nbfolds), std::end(require_nbfolds), this->problemType) != std::end(require_nbfolds);
	if (exists == true && this->nb_folds < 0)
	{
		cout << "ERROR: the chosen method requires nb_folds to be supplied by the user." << endl;
		return;
	}

	// Number of breakpoints - not permitted
	ProblemType disallow_nbbreakpts[5] = {ERM, ERM_l0, ERM_l1, BL, BL_SS};
	exists = std::find(std::begin(disallow_nbbreakpts), std::end(disallow_nbbreakpts), this->problemType) != std::end(disallow_nbbreakpts);
	if (exists == true && this->nb_breakpoints >= 0)
	{
		cout << "ERROR: the chosen method does not support number of breakpoints nb_breakpoints." << endl;
		return;
	}
	// Number of breakpoints - required
	ProblemType require_nbbreakpts[4] = {GS_ERM_l0, GS_ERM_l1, GS_SS_ERM_l0, GS_SS_ERM_l1};
	exists = std::find(std::begin(require_nbbreakpts), std::end(require_nbbreakpts), this->problemType) != std::end(require_nbbreakpts);
	if (exists == true && this->nb_breakpoints < 0)
	{
		cout << "ERROR: the chosen method requires nb_breakpoints to be supplied by the user." << endl;
		return;
	}

	// Regularization parameter - not permitted
	ProblemType disallow_lambda[6] = {BL, BL_SS, GS_ERM_l0, GS_ERM_l1, GS_SS_ERM_l0, GS_SS_ERM_l1};
	exists = std::find(std::begin(disallow_lambda), std::end(disallow_lambda), this->problemType) != std::end(disallow_lambda);
	if (exists == true && this->regularization_param > 0)
	{
		cout << "ERROR: the chosen method does not support the regulatization parameter lambda." << endl;
		return;
	}

	// Split size - not permitted
	ProblemType disallow_split_size[6] = {ERM, ERM_l0, ERM_l1, BL, GS_ERM_l0, GS_ERM_l1};
	exists = std::find(std::begin(disallow_split_size), std::end(disallow_split_size), this->problemType) != std::end(disallow_split_size);
	if (exists == true && this->split_size > 0)
	{
		cout << "ERROR: the chosen method does not support the split size parameter." << endl;
		return;
	}

	// Split features - not permitted
	ProblemType disallow_split_feat[6] = {ERM, ERM_l0, ERM_l1, BL, GS_ERM_l0, GS_ERM_l1};
	exists = std::find(std::begin(disallow_split_feat), std::end(disallow_split_feat), this->problemType) != std::end(disallow_split_feat);
	if (exists == true && this->split_feat != 1)
	{
		cout << "ERROR: the chosen method does not support the split features parameter." << endl;
		return;
	}

	//Input
	this->command_ok = true;
}

commandline::~commandline() {}


string commandline::get_instance_path()
{
	return instance_path;
}

string commandline::get_instance_filename()
{
    string to_parse = this->instance_path;

    char caractere1 = '/';
	char caractere2 = '\\';

	int position = to_parse.find_last_of(caractere1);
	int position2 = to_parse.find_last_of(caractere2);
	if (position2 > position)
		position = position2;

	if (position != -1)
		return to_parse.substr(position + 1);
	else
		return to_parse;
}

string commandline::get_output_path()
{
	return output_path;
}

bool commandline::is_valid()
{
	return command_ok;
}

double commandline::get_timeLimit()
{
	return timeLimit;
}

int commandline::get_nbThreads()
{
	return nbThreads;
}

double commandline::get_backorder_cost()
{
	return backorder_cost;
}

double commandline::get_holding_cost()
{
	return holding_cost;
}

ProblemType commandline::get_problem_type()
{
	return problemType;
}

string commandline::get_str_problem_type()
{
	return strProblemType;
}

double commandline::get_train_val_split()
{
	return train_val_split;
}

int commandline::get_split_size()
{
	return split_size;
}

double commandline::get_split_feat()
{
	return split_feat;
}

double commandline::get_regularization_param()
{
	return regularization_param;
}

int commandline::get_nb_folds()
{
	return nb_folds;
}

int commandline::get_nb_breakpoints()
{
	return nb_breakpoints;
}

double commandline::get_informative_factor()
{
	return informativeFactor;
}

bool commandline::get_set_value_z()
{
	return set_value_z;
}
