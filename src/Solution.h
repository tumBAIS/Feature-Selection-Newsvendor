#ifndef SOLUTION_H
#define SOLUTION_H

#include <algorithm>    // std::max
#include <iomanip>
#include <numeric>      // std::iota
#include "Pb_Data.h"

template<typename T>
void exportVector(const vector<T>& vec, std::ofstream& file, string varname, char sep);

template<typename T>
void exportMatrix(const vector<vector<T>>& mat, std::ofstream& file, string varname, char sep);

bool isSolOpt(int solver_status);

class Solution
{
public:
    // data for which this solution was built
    std::shared_ptr<Pb_Data> myData;

    // number of columns
    int numCols;

    // number of rows
    int numRows;

    // solution status (from CPLEX)
    int solverStatus;

    // status string
    string strStatus;

    // solution objective value (from CPLEX)
    double objVal;

    // lower bound value (from CPLEX)
    double lowerBoundVal;

    // node count (from CPLEX)
    int nodeCount;

    // MIP relative gap (from CPLEX)
    double mipGap;

    // Solution array
    vector<double> solutionArray;

    // Objective cost on training set
    double trainCost;

    // Objective cost on validation set
    double valCost;

    // Objective cost on training + validation set
    double trainValCost;

    // Objective cost on test set
    double testCost;

    // Optimal objective cost given features on the training set
    double trainCostOpt;

    // Optimal objective cost given features on the validation set
    double valCostOpt;

    // Optimal objective cost given features on the training + validation set
    double trainValCostOpt;

    // Optimal objective cost given features on the test set
    double testCostOpt;

    // solution time
    double solution_time;

    // Solution - model coefficients
    vector<double> solBeta;  

    /* METHODS TO TEST AND EXPORT THE FINAL SOLUTION */
    // Generate output filename where the solution is exported to
    string getOutputFilename(string extension);

    string doubleToString(double inputVal);

    // Creates a copy of input file
    void copyInputFile();

    // Constructor
    Solution(std::shared_ptr<Pb_Data> myData);

    static std::shared_ptr<Solution> createSolution(std::shared_ptr<Pb_Data> myData);

    // Calculate objective cost
    double calculateTrainingCost(const vector<double>& beta);
    double calculateValCost(const vector<double>& beta);
    double calculateTrainValCost(const vector<double>& beta);
    double calculateTestCost(const vector<double>& beta);

    // Update solution
    virtual void update(int numCols, int numRows, int solverStatus, string strStatus, double objVal, 
                        double lowerBoundVal, int nodeCount, double mipGap, double *solution);

    // Update solution vectors
    virtual void updateSolutionVectors() = 0;

    // Method to display a solution
    void displaySolution();

    // Export solution to a txt file
    void exportSolution(string ext);

    // Export solution arrays to a txt file
	virtual void exportSolutionArrays(std::ofstream& file) = 0;
};


class SolutionBilevel : public Solution
{
public:

	SolutionBilevel(std::shared_ptr<Pb_Data> myData) : Solution(myData) {}

    int sizeU, sizeO, sizeBeta, sizeMu, sizeGamma, sizeZ, sizeSoln;

    vector<double> solU; 
    vector<double> solO;         
    vector<double> solMu;
    vector<double> solGamma;
    vector<int> solZ;

    // Update solution vectors
    void updateSolutionVectors() override;

    // Export solution to a txt file
	void exportSolutionArrays(std::ofstream& file) override;
};

class SolutionBilevelShuffleSplit : public Solution
{
public:

    SolutionBilevelShuffleSplit(std::shared_ptr<Pb_Data> myData) : Solution(myData) {}

    vector<vector<double>> solU; 
    vector<vector<double>> solO;         
    vector<vector<double>> solBetaMatrix;  
    vector<double> solMu;
    vector<double> solGamma;
    vector<int> solZ;

    // Update solution vectors
    void updateSolutionVectors() override;

    // Export solution to a txt file
	void exportSolutionArrays(std::ofstream& file) override;
};

class SolutionERM : public Solution
{
public:

	SolutionERM(std::shared_ptr<Pb_Data> myData) : Solution(myData) {}

    vector<double> solU; 
    vector<double> solO;

    // Update solution vectors
    void updateSolutionVectors() override;

    // Export solution to a txt file
	void exportSolutionArrays(std::ofstream& file) override;
};


class SolutionERM_l0 : public Solution
{
public:

	SolutionERM_l0(std::shared_ptr<Pb_Data> myData) : Solution(myData) {}

    vector<double> solU; 
    vector<double> solO;         
    vector<int>    solZ;

    // Update solution vectors
    void updateSolutionVectors() override;

    // Export solution to a txt file
	void exportSolutionArrays(std::ofstream& file) override;
};

class SolutionERM_l1 : public Solution
{
public:

	SolutionERM_l1(std::shared_ptr<Pb_Data> myData) : Solution(myData) {}

    vector<double> solU; 
    vector<double> solO;         
    vector<double> solBeta_pos;
    vector<double> solBeta_neg;

    // Update solution vectors
    void updateSolutionVectors() override;

    // Export solution to a txt file
	void exportSolutionArrays(std::ofstream& file) override;
};


class SolutionSubProblem : public Solution
{
public:

    SolutionSubProblem(std::shared_ptr<Pb_Data> myData) : Solution(myData) {}

    static std::shared_ptr<SolutionSubProblem> createSolutionSP(std::shared_ptr<Pb_Data> myData);

    // Update solution vectors
    virtual void updateSolutionVectors() = 0;

    // Export solution to a txt file
	virtual void exportSolutionArrays(std::ofstream& file) = 0;
};

class SolutionSubProblem_ERM_l0 : public SolutionSubProblem
{
public:

    SolutionSubProblem_ERM_l0(std::shared_ptr<Pb_Data> myData) : SolutionSubProblem(myData) {}   

    vector<double> solU; 
    vector<double> solO;         
    vector<int>    solZ;

    // Update solution vectors
    void updateSolutionVectors() override;

    // Export solution to a txt file
	void exportSolutionArrays(std::ofstream& file) override;
};



class SolutionSubProblem_ERM_l1 : public SolutionSubProblem
{
public:

    SolutionSubProblem_ERM_l1(std::shared_ptr<Pb_Data> myData) : SolutionSubProblem(myData) {}   

    vector<double> solU; 
    vector<double> solO;     
    vector<double> solBeta_pos;
    vector<double> solBeta_neg;
    vector<int> solZ;
    
    // Update solution vectors
    void updateSolutionVectors() override;

    // Export solution to a txt file
	void exportSolutionArrays(std::ofstream& file) override;
};

class SolutionGridSearch : public Solution
{
public:

    SolutionGridSearch(std::shared_ptr<Pb_Data> myData) : Solution(myData) {}

    vector<std::shared_ptr<SolutionSubProblem>> spSolutions;

    std::shared_ptr<SolutionSubProblem> finalSolution;

    vector<double> lambdaVector;     // Regularization parameter break-point values
    vector<double> trainCostVector; // Training cost for each break-point
    vector<double> valCostVector;   // Validation cost for each break-point

    // Update solution vectors
    virtual void updateSolutionVectors() = 0;

    // Export solution to a txt file
	virtual void exportSolutionArrays(std::ofstream& file) = 0;
};

class SolutionGridSearch_ERM_l0 : public SolutionGridSearch
{
public:

    SolutionGridSearch_ERM_l0(std::shared_ptr<Pb_Data> myData) : SolutionGridSearch(myData) {}

    int sizeU, sizeO, sizeBeta, sizeZ, sizeSoln;

    vector<double> solU; 
    vector<double> solO;
    vector<int>    solZ;

    // Update solution vectors
    void updateSolutionVectors() override;

    // Export solution to a txt file
	void exportSolutionArrays(std::ofstream& file) override;
};


class SolutionGridSearch_ERM_l1 : public SolutionGridSearch
{
public:

    SolutionGridSearch_ERM_l1(std::shared_ptr<Pb_Data> myData) : SolutionGridSearch(myData) {}

    int sizeU, sizeO, sizeBeta, sizeSoln;

    vector<double> solU; 
    vector<double> solO;         
    vector<double> solBeta_pos;
    vector<double> solBeta_neg;

    // Update solution vectors
    void updateSolutionVectors() override;

    // Export solution to a txt file
	void exportSolutionArrays(std::ofstream& file) override;
};


class SolutionGridSearchCV : public Solution
{
public:

    SolutionGridSearchCV(std::shared_ptr<Pb_Data> myData) : Solution(myData) {}

    vector<vector<std::shared_ptr<SolutionSubProblem>>> spSolutions;

    std::shared_ptr<SolutionSubProblem> finalSolution;

    vector<double> lambdaVector;     // Regularization parameter break-point values
    vector<vector<double>> trainCostVector; // Training cost for each break-point
    vector<vector<double>> valCostVector;   // Validation cost for each break-point
    vector<double> avgValCostVector;

    // Update solution vectors
    virtual void updateSolutionVectors() = 0;

    // Export solution to a txt file
	virtual void exportSolutionArrays(std::ofstream& file) = 0;
};


class SolutionGridSearchCV_ERM_l0 : public SolutionGridSearchCV
{
public:

    SolutionGridSearchCV_ERM_l0(std::shared_ptr<Pb_Data> myData) : SolutionGridSearchCV(myData) {}

    int sizeU, sizeO, sizeBeta, sizeZ, sizeSoln;

    vector<double> solU; 
    vector<double> solO;       
    vector<int>    solZ;  

    // Update solution vectors
    void updateSolutionVectors() override;

    // Export solution to a txt file
	void exportSolutionArrays(std::ofstream& file) override;
};

class SolutionGridSearchCV_ERM_l1 : public SolutionGridSearchCV
{
public:

    SolutionGridSearchCV_ERM_l1(std::shared_ptr<Pb_Data> myData) : SolutionGridSearchCV(myData) {}

    int sizeU, sizeO, sizeBeta, sizeSoln;

    vector<double> solU; 
    vector<double> solO;  
    vector<double> solBeta_pos;
    vector<double> solBeta_neg;     

    // Update solution vectors
    void updateSolutionVectors() override;

    // Export solution to a txt file
	void exportSolutionArrays(std::ofstream& file) override;
};

#endif
