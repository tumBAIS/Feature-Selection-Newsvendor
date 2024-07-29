#ifndef SOLVER_H
#define SOLVER_H

#include <time.h> 
#include <stdlib.h>
#include "Pb_Data.h"
#include "Solution.h"

vector<double> linspace(const double start, const double end, const int num_in);

vector<double> logspace(const double start, const double end, 
						const int num_in, const double base);

template<typename T>
void print_vector(const vector<T>& vec);

template<typename T>
void print_vector(const vector<T>& vec)
{
    std::cout << "size: " << vec.size() << std::endl;
    for (T d : vec)
        std::cout << d << " ";
    std::cout << std::endl;
}

// Solver for the newsvendor problem with feature selection
class Solver 
{
protected:
	// Parameters of the problem
	std::shared_ptr<Pb_Data> myData;

	// Parameters of the problem
	std::shared_ptr<Solution> mySolution = NULL;

	void *env = NULL;
	void *model = NULL;

	// Converts solver status to human-readable string
	string getSolverStatusString(int modelstatus);
	
	// Create columns related to the beta variables
	virtual int addBetaVariables(void *env, void *model, int& size);

	// Add constraints related to underage costs
	virtual int addUnderageConstrs(void *env, void *model);

	// Add constraints related to overage costs
	virtual int addOverageConstrs(void *env, void *model);

	// Add indicator constraints related to the beta and z variables
	virtual int addBetaIndConstrs(void *env, void *model, const int startBeta, const int startZ);

	int solverAddCols(void *env, void *model, int ccnt, double *obj, double *lb, double *ub, char *vtype, char **colname);
	int solverAddRows(void *env, void *model, int rcnt, int nzcnt, double *rhs, char *sense, int *rmatbeg, int *rmatind, double *rmatval, char **rowname);
	int solverAddIndConstr(void *env, void *model, int indvar, int complemented, int nzcnt, double rhs, int sense, int *linind, double *linval, char *indname_str);

	int initializeModel(int logToConsole, char const *modelname);

	int solverWriteModel(void *env, void *model, char const *filename);

	int solverGetModelDimensions(void *env, void *model, int& numcols, int& numlinconstrs, int& numindconstrs, int& numrows);

	int solverOptimize(void *env, void *model, ModelType model_type);

	int solverRetrieveSolution(void *env, void *model, int *modelstatus, double *objval, double *bestobjval, double *mipgap, double *nodecount, double *solution, int sizeVars);

	int quit_solver(void *env, void *model);

public:
    // Constructor common to all Solver solvers
	Solver(std::shared_ptr<Pb_Data> myData, std::shared_ptr<Solution> mySolution) : myData(myData), mySolution(mySolution) {}

	static std::unique_ptr<Solver> createSolver(std::shared_ptr<Pb_Data> myData, std::shared_ptr<Solution> mySolution);

    // Returns 0 if solved properly
	virtual int solve() = 0;

	// Virtual destructor (needs to stay virtual otherwise inherited destructors will not be called)
	virtual ~Solver(void) {};
};

class SolverBilevel : public Solver
{
public:

	SolverBilevel(std::shared_ptr<Pb_Data> myData, std::shared_ptr<Solution> mySolution) : Solver(myData, mySolution) {}

	// Create columns related to the u variables
	int addUnderageVariables(void *env, void *model, int& size);

	// Create columns related to the o variables
	int addOverageVariables(void *env, void *model, int& size);

	// Create columns related to the mu variables
	int addDualMuVariables(void *env, void *model, int& size);

	// Create columns related to the gamma variables
	int addDualGammaVariables(void *env, void *model, int& size);

	// Add constraint related to the lower-level problem's optimality condition
	int addFollowerOptConstr(void *env, void *model);

	// Add constraints related to dual variable mu
	int addDualMuConstrs(void *env, void *model);

	// Add constraints related to dual variable gamma
	int addDualGammaConstrs(void *env, void *model);

	// Add indicator constraints related to the dual variables mu and gamma
	int addDualIndConstrs(void *env, void *model);

	int solve() override;
};

class SolverBilevelShuffleSplit : public Solver
{
public:

	SolverBilevelShuffleSplit(std::shared_ptr<Pb_Data> myData, std::shared_ptr<Solution> mySolution) : Solver(myData, mySolution) {}

	// Create columns related to the u variables
	int addUnderageVariables(void *env, void *model, int& size);

	// Create columns related to the o variables
	int addOverageVariables(void *env, void *model, int& size);

	// Create columns related to the beta variables
	int addBetaVariables(void *env, void *model, int& size) override;

	// Create columns related to the mu variables
	int addDualMuVariables(void *env, void *model, int& size);

	// Create columns related to the gamma variables
	int addDualGammaVariables(void *env, void *model, int& size);

	// Create columns related to the z variables
	int addZVariables(void *env, void *model, int& size);

	// Add constraints related to underage costs
	int addUnderageConstrs(void *env, void *model) override;

	// Add constraints related to overage costs
	int addOverageConstrs(void *env, void *model) override;

	// Add indicator constraints related to the beta and z variables
	int addBetaIndConstrs(void *env, void *model, const int startBeta, const int startZ) override;

	// Add constraint related to the lower-level problem's optimality condition
	int addFollowerOptConstr(void *env, void *model);

	// Add constraints related to dual variable mu
	int addDualMuConstrs(void *env, void *model, const int startMu);

	// Add constraints related to dual variable gamma
	int addDualGammaConstrs(void *env, void *model, const int startGamma);

	// Add indicator constraints related to the dual variables mu and gamma
	int addDualIndConstrs(void *env, void *model);

	int solve() override;
};

class SolverERM : public Solver
{
public:

	SolverERM(std::shared_ptr<Pb_Data> myData, std::shared_ptr<Solution> mySolution) : Solver(myData, mySolution) {}

	// Create columns related to the u variables
	int addUnderageVariables(void *env, void *model, int& size);

	// Create columns related to the o variables
	int addOverageVariables(void *env, void *model, int& size);

	virtual int solve() override;
};

class SolverERM_l0 : public SolverERM
{
public:

	SolverERM_l0(std::shared_ptr<Pb_Data> myData, std::shared_ptr<Solution> mySolution) : SolverERM(myData, mySolution) {}

	int solve() override;
};

class SolverERM_l1 : public SolverERM
{
public:

	SolverERM_l1(std::shared_ptr<Pb_Data> myData, std::shared_ptr<Solution> mySolution) : SolverERM(myData, mySolution) {}

	int solve() override;
};

class SolverSubProblem : public Solver
{
public:
	SolverSubProblem(std::shared_ptr<Pb_Data> myData, std::shared_ptr<Solution> mySolution) : Solver(myData, mySolution) {}

	int spIndex = -1;

	// Create columns related to the u variables
	int addUnderageVariables(void *env, void *model, int& size);

	// Create columns related to the o variables
	int addOverageVariables(void *env, void *model, int& size);

	static std::unique_ptr<SolverSubProblem> createSolverSP(std::shared_ptr<Pb_Data> myData, std::shared_ptr<Solution> mySolution);

	virtual int solve() = 0;
};

class SolverSubProblem_ERM_l0 : public SolverSubProblem
{
public:
	SolverSubProblem_ERM_l0(std::shared_ptr<Pb_Data> myData, std::shared_ptr<Solution> mySolution) : SolverSubProblem(myData, mySolution) {}

	int addUnderageConstrs(void *env, void *model) override;

	int addOverageConstrs(void *env, void *model) override;

	int solve() override;
};

class SolverSubProblem_ERM_l1 : public SolverSubProblem
{
public:
	SolverSubProblem_ERM_l1(std::shared_ptr<Pb_Data> myData, std::shared_ptr<Solution> mySolution) : SolverSubProblem(myData, mySolution) {}

	int addUnderageConstrs(void *env, void *model) override;

	int addOverageConstrs(void *env, void *model) override;

	int solve() override;
};

class SolverGridSearch : public Solver
{
public:

	SolverGridSearch(std::shared_ptr<Pb_Data> myData, std::shared_ptr<Solution> mySolution) : Solver(myData, mySolution) {}

	int solve() override;

	vector<std::unique_ptr<SolverSubProblem>> spSolvers;

	std::unique_ptr<SolverSubProblem> finalSolver = NULL;
};


class SolverGridSearchCV : public Solver
{
public:

	SolverGridSearchCV(std::shared_ptr<Pb_Data> myData, std::shared_ptr<Solution> mySolution) : Solver(myData, mySolution) {}

	int solve() override;

	vector<vector<std::unique_ptr<SolverSubProblem>>> spSolvers;

	std::unique_ptr<SolverSubProblem> finalSolver = NULL;
};

#endif
