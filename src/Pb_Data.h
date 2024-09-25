#ifndef DATA_H
#define DATA_H

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <cassert>
#include <vector>
#include <sstream>
#include <utility>
#include <math.h>
#include <algorithm>
#include <exception>
#include <memory>

#define ILOSTLBEGIN
#include <ilcplex/ilocplex.h>

using namespace std;

enum ProblemType
{
    BL,                 // Bilevel Feature Selection (BFS)
    BL_SS,              // BFS with random permutations cross-validation a.k.a. Shuffle & Split
    ERM,                // Empirical Risk Minimization (without Regularization)
    ERM_l0,             // Empirical Risk Minimization with l0-norm regularization
    ERM_l1,             // Empirical Risk Minimization with l1-norm regularization
    SP_ERM_l0,          // Subproblem ERM with l0-norm regularization (trained on the training set)
    SP_ERM_l1,          // Subproblem ERM with l1-norm regularization (trained on the training set)
    GS_ERM_l0,          // Grid search for ERM with l0-norm regularization
    GS_ERM_l1,          // Grid search for ERM with l1-norm regularization
    GS_SS_ERM_l0,       // Grid search with Shuffle & Split cross-validation for ERM with l0-norm regularization
    GS_SS_ERM_l1        // Grid search with Shuffle & Split cross-validation for ERM with l0-norm regularization
};

enum ModelType
{
    MIP_PROBLEM,
    LP_PROBLEM
};

class Pb_Data
{
public:
    /* PROBLEM DATA */

    // address of the problem instance, including instance name
    string instance_path;

    // it's the basename of the pathToInstance
    string instance_filename;

    // instance name (without extension)
    string instance_name;

    // instance base folder
    string instance_basefolder;

    // extension of instance file
    const string train_data_extension = ".train";
    const string test_data_extension = ".test";

    // Test data path
    string test_data_path;

    // Test data filename
    string test_data_filename;

    // output path (where results will be saved)
    string output_path;

    // model path
    bool save_model = false;

    // whether to save full solution (including solution vector for u, o, mu, gamma, etc) or not
    bool save_full_soln = false;

    // back-ordering cost
    double backOrderCost;

    // holding cost
    double holdingCost;

    // type of problem solved
    ProblemType problemType;

    // string describing problem type
    string strProblemType;

    // Train-Validation split
    double trainValSplit;

    // Regularization parameter (for ERM models)
    double regularizationParam;

    // Number of folds for cross validation
    int nbFolds;

    // Separator
    char sep = '\t';

    // number of samples
    int nbSamples;

    // number of features
    int nbFeatures;

    // number of informative features
    int nbInformativeFeatures;

    // number of features to be active in the solution
    int numActiveFeatures = -1;

    // number of breakpoints (for grid search)
    int nbBreakpoints;

    // number of training samples
    int nbTrainSamples;

    // number of validation samples
    int nbValSamples;

    // number of test samples
    int nbTestSamples;

    // Create a vector to keep track of the hold-out validation split (training / validation split)
    vector<int> holdoutSplit;

    // Create a vector to keep track of the cross validation split
    vector<vector<int>> crossValSplit;

    // Nominal size of each split as given by the user
    int nominalSplitSize;

    // Size of each split actually used (including training + validation sets)
    int splitSize;

    // Size of training set for each split
    int splitTrainSize;

    // Size of validation set for each split
    int splitValSize;
    
    // Number of features to be used in each split
    int splitNbFeatures;

    // Indicates which features are active in each split for the BL Shuffle & Split
    vector<vector<int>> activeFeatures;

    // Base fold size (this corresponds to the validation set size, except possibly for the last validation set)
    int baseFoldSize;

    // Create a vector of <string, int vector> pairs to store the ground truth parameters
    std::vector<std::pair<std::string, std::vector<double>>> ground_truth_params;

    // Create a vector of <string, int vector> pairs to store the data set
    std::vector<std::pair<std::string, std::vector<double>>> dataset;

    // Create a vector for the ground truth beta parameter vector
    std::vector<double> ground_truth_beta;

    // Create a vector for the demand observations
    std::vector<double> demand;

    // Create a matrix for the feature data
    std::vector<std::vector<double>> feature_data;

    // Test data set
    std::vector<std::pair<std::string, std::vector<double>>> test_dataset;

    // Demand observations of the test data set
    std::vector<double> test_demand;

    // Feature data of the test data set
    std::vector<std::vector<double>> test_feature_data;

    // Starting time of the optimization
    clock_t time_StartOpt;

    // End time of the optimization
    clock_t time_EndOpt;

    // number of threads for CPLEX solvers
    int nbThreads;

    // solver time limit
    double timeLimit;

    // Display table
    void display_table(std::vector<std::pair<std::string, 
                       std::vector<double>>> table);

    // Read instance from file
    void read_data(string path);

    // Read test data from file
    void read_test_data(string path);

    // Reset hold-out split
    void resetSplit();

    // Modify hold-out split
    void modifySplit(const vector<int>& newHoldoutSplit);

    // Constructor
    Pb_Data(string pathToInstance, string instanceFilename, string outputPath, 
            double backOrderCost, double holdingCost, ProblemType problemType, 
            string strProblemType, double trainValSplit, int splitSize, 
            double regularizationParam, int nbFolds, int nbBreakpoints, 
            int nbThreads, double timeLimit);
};

#endif