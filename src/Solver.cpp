#include "Solver.h"

std::unique_ptr<Solver> Solver::createSolver(std::shared_ptr<Pb_Data> myData, std::shared_ptr<Solution> mySolution)
{
    if (myData->problemType == BL)
        return std::make_unique<SolverBilevel> (myData, mySolution);

    else if (myData->problemType == BL_SS)
        return std::make_unique<SolverBilevelShuffleSplit> (myData, mySolution);

    else if (myData->problemType == ERM)
        return std::make_unique<SolverERM> (myData, mySolution);

    else if (myData->problemType == ERM_l0)
        return std::make_unique<SolverERM_l0> (myData, mySolution);

    else if (myData->problemType == ERM_l1)
        return std::make_unique<SolverERM_l1> (myData, mySolution);

    else if (myData->problemType == SP_ERM_l0)
        return std::make_unique<SolverSubProblem_ERM_l0> (myData, mySolution);

    else if (myData->problemType == SP_ERM_l1)
        return std::make_unique<SolverSubProblem_ERM_l1> (myData, mySolution);    

    else if ((myData->problemType == GS_ERM_l0) || 
             (myData->problemType == GS_ERM_l1))
        return std::make_unique<SolverGridSearch> (myData, mySolution);

    else if ((myData->problemType == GS_SS_ERM_l0) || 
             (myData->problemType == GS_SS_ERM_l1))
        return std::make_unique<SolverGridSearchCV> (myData, mySolution);

    else
    {   
        std::cerr << "Failed to create solver structure: no solver matching the input could be found" << std::endl;
        return NULL;
    }
}

std::unique_ptr<SolverSubProblem> SolverSubProblem::createSolverSP(std::shared_ptr<Pb_Data> myData, std::shared_ptr<Solution> mySolution)
{
    if ((myData->problemType == GS_ERM_l0) ||
        (myData->problemType == GS_SS_ERM_l0))
        return std::make_unique<SolverSubProblem_ERM_l0> (myData, mySolution);

    else if ((myData->problemType == GS_ERM_l1) ||
             (myData->problemType == GS_SS_ERM_l1))
        return std::make_unique<SolverSubProblem_ERM_l1> (myData, mySolution); 

    else 
    {
        std::cerr << "Failed to create solver structure for subproblem: no solver matching the input could be found" << std::endl;
        return NULL;
    }
}

string Solver::getSolverStatusString(int lpstat)
{
    // Convert LP status to string (https://www.tu-chemnitz.de/mathematik/discrete/manuals/cplex/doc/refman/html/appendixB.html)
    string stat_str;
    if (lpstat == CPXMIP_OPTIMAL) 
        stat_str.assign("OPTIMAL");
    else if (lpstat == CPXMIP_OPTIMAL_TOL)
        stat_str.assign("OPTIMAL_TOL");
    else if (lpstat == CPX_STAT_OPTIMAL) 
        stat_str.assign("STAT_OPTIMAL");
    else if (lpstat == CPXMIP_INFEASIBLE)
        stat_str.assign("INFEASIBLE");
    else if (lpstat == CPXMIP_TIME_LIM_FEAS) 
        stat_str.assign("TIME_LIM_FEAS");
    else if (lpstat == CPXMIP_TIME_LIM_INFEAS) 
        stat_str.assign("TIME_LIM_INFEAS");         
    else if (lpstat == CPX_STAT_UNBOUNDED) 
        stat_str.assign("STAT_UNBOUNDED");
    else if (lpstat == CPX_STAT_INFEASIBLE) 
        stat_str.assign("STAT_INFEASIBLE");
    else if (lpstat == CPX_STAT_ABORT_DETTIME_LIM)
        stat_str.assign("STAT_ABORT_DETTIME_LIM");
    else if (lpstat == CPXMIP_ABORT_INFEAS)
        stat_str.assign("MIP_ABORT_INFEAS");
    
    else
        stat_str.assign("?");
    
    return stat_str;
}

vector<double> linspace(const double start, const double end, const int num_in)
{
    vector<double> linspaced;
    double num = (double) num_in;

    if (num == 0) { return linspaced; }
    if (num == 1) 
    {
        linspaced.push_back(start);
        return linspaced;
    }

    double delta = (end - start) / (num - 1);

    for(int i=0; i < num-1; ++i)
    {
        linspaced.push_back(start + delta * i);
    }
    linspaced.push_back(end);   // I want to ensure that start and end
                                // are exactly the same as the input
    return linspaced;
}


vector<double> logspace(const double start, const double end, 
                        const int num_in, const double base)
{
    vector<double> logspaced;
    vector<double> linspaced = linspace(start, end, num_in);

    for (double d : linspaced) {
        logspaced.push_back( pow(base, d) );
    }
    return logspaced;
}

int Solver::quit_solver()
{
    int status = 0;
    char errbuf[CPXMESSAGEBUFSIZE];

    // Free the problem as allocated by CPXcreateprob and CPXreadcopyprob, if necessary
    if (model != NULL) {
        int xstatus = CPXfreeprob(env, &model);
        if (!status) status = xstatus;
        if (status) {
            std::cerr << "Failed to free memory for problem: " 
                << CPXgeterrorstring(env, status, errbuf) << std::endl;
        }
    }

    // Free the CPLEX environment, if necessary
    if (env != NULL) {
        int xstatus = CPXcloseCPLEX(&env);
        if (!status) status = xstatus;
        if (status) {
            std::cerr << "Failed to close CPLEX: " 
                << CPXgeterrorstring(env, status, errbuf) << std::endl;
        }
    }
    return status;
}

int Solver::initializeModel(int logToConsole, char const *modelname)
{
    int status = 0;

    this->env = CPXopenCPLEX(&status); // Open CPLEX environment
    if (status) return status;

    CPXsetintparam(this->env, CPXPARAM_ScreenOutput, logToConsole); // Switching ON the display
    CPXsetintparam(this->env, CPXPARAM_Read_DataCheck, CPX_DATACHECK_WARN); // Print warnings
    CPXsetintparam(this->env, CPX_PARAM_THREADS, myData->nbThreads);		  // number of threads
    CPXsetdblparam(this->env, CPX_PARAM_TILIM, myData->timeLimit); // sets time limit for the solver.

    this->model = CPXcreateprob(this->env, &status, modelname); // Create LP problem as a container
    return status;
}

int Solver::solverAddCols(int ccnt, double *obj, double *lb, double *ub, char *vtype, char **colname)
{
    int status = 0;

    char errbuf[CPXMESSAGEBUFSIZE];
    status = CPXnewcols(env, model, ccnt, obj, lb, ub, vtype, colname);
    if (status) {
        std::cerr << "Failed to add variables: " 
            << CPXgeterrorstring(env, status, errbuf) << std::endl;
    }
    return status;
}

int Solver::solverAddRows(int rcnt, int nzcnt, double *rhs, char *sense, int *rmatbeg, int *rmatind, double *rmatval, char **rowname)
{
    int status = 0;

    status = CPXaddrows(env, model, 0, rcnt, nzcnt, rhs, sense, rmatbeg, rmatind, rmatval, NULL, rowname);
    return status;
}

int Solver::solverAddIndConstr(int indvar, int complemented, int nzcnt, double rhs, int sense, int *linind, double *linval, char *indname_str)
{
    int status = 0;

    status = CPXaddindconstr(env, model, indvar, complemented, nzcnt, rhs, sense, linind, linval, indname_str);
    return status;
}

int Solver::solverWriteModel(char const *filename)
{
    int status = 0;

    status = CPXwriteprob(env, model, filename, NULL); // Exporting the model
    return status;
}

int Solver::solverGetModelDimensions(int& numcols, int& numlinconstrs, int& numindconstrs, int& numrows)
{
    int status = 0;
    numcols = CPXgetnumcols(env, model); // Get number of columns
    numlinconstrs = CPXgetnumrows(env, model); // Get number of rows 
    numindconstrs = CPXgetnumindconstrs(env, model); // Get number of indicator constraints
    numrows = numlinconstrs + numindconstrs;
    return status;
}

int Solver::solverOptimize(ModelType model_type)
{
    int status = 0;

    char errbuf[CPXMESSAGEBUFSIZE];
    if (model_type == LP_PROBLEM) {
        status = CPXlpopt(env, model);
    } else if (model_type == MIP_PROBLEM) {
        status = CPXmipopt(env, model);
    }
    if (status) {
        std::cerr << "Failed to optimize: " 
            << CPXgeterrorstring(env, status, errbuf) << std::endl;
    }
    return status;
}

int Solver::solverRetrieveSolution(int *modelstatus, double *objval, double *bestobjval, double *mipgap, double *nodecount, double *solution, int sizeVars)
{
    int status = 0;

    char errbuf[CPXMESSAGEBUFSIZE];
    status = CPXsolution(env, model, modelstatus, objval, solution, NULL, NULL, NULL); // Get solution array
    if (status) {
        std::cerr << "Failed to retrieve solution: " 
            << CPXgeterrorstring(env, status, errbuf) << std::endl;
    }

    if (nodecount){
        *nodecount = CPXgetnodecnt(env, model); // Get node count
    }

    if (mipgap) {
        status = CPXgetmiprelgap(env, model, mipgap); // Get MIP gap
        if (status) {
            std::cerr << "Failed to retrieve MIP gap: " 
                << CPXgeterrorstring(env, status, errbuf) << std::endl;
        }
    }

    if (bestobjval) {
        status = CPXgetbestobjval(env, model, bestobjval); // Get best objective value
        if (status) {
            std::cerr << "Failed to retrieve best obj val: " 
                << CPXgeterrorstring(env, status, errbuf) << std::endl;
        }
    }
    return status;
}


int SolverBilevel::addUnderageVariables(int& size)
{
    // Create columns related to the u variables
    int sizeU = myData->nbSamples;
	double *costU = new double[sizeU];
	char **namesU = new char *[sizeU];
	for (int i = 0; i < myData->nbSamples; i++)
	{
        if (i < myData->nbTrainSamples)
            costU[i] = 0;
        else
            costU[i] = (myData->backOrderCost / (double) myData->nbValSamples);
        namesU[i] = new char[100];
		sprintf(namesU[i], "u_%d", i);
	}
    int status = solverAddCols(sizeU, costU, NULL, NULL, NULL, namesU);
    
    delete[] costU;
    for (int i = 0; i < sizeU; i++)
		delete[] namesU[i];
    delete[] namesU;

    size = sizeU; // return number of variables added to the model
    return status;
}

int SolverBilevel::addOverageVariables(int& size)
{
    // Create columns related to the o variables
    int sizeO = myData->nbSamples;
    double *costO = new double[sizeO];
	char **namesO = new char *[sizeO];
	for (int i = 0; i < myData->nbSamples; i++)
	{
        if (i < myData->nbTrainSamples)
            costO[i] = 0;
        else
            costO[i] = (myData->holdingCost / (double) myData->nbValSamples);
        namesO[i] = new char[100];
		sprintf(namesO[i], "o_%d", i);
	}
    int status = solverAddCols(sizeO, costO, NULL, NULL, NULL, namesO);

    delete[] costO;
    for (int i = 0; i < sizeO; i++)
		delete[] namesO[i];
    delete[] namesO;

    size = sizeO; // return number of variables added to the model
    return status;
}

int SolverBilevel::addDualMuVariables(int& size)
{
    // Create columns related to the mu variables
    int sizeMu = myData->nbTrainSamples;
    char **namesMu = new char *[sizeMu];
    double *lbMu = new double[sizeMu];
    double *ubMu = new double[sizeMu];
    for (int i = 0; i < myData->nbTrainSamples; i++)
    {
        lbMu[i] = -CPX_INFBOUND;
        ubMu[i] = 0.0;
        namesMu[i] = new char[100];
        sprintf(namesMu[i], "mu_%d", i);
    }
    int status = solverAddCols(sizeMu, NULL, lbMu, ubMu, NULL, namesMu);

    for (int i = 0; i < sizeMu; i++)
        delete[] namesMu[i];
    delete[] namesMu;
    delete[] lbMu;
    delete[] ubMu;

    size = sizeMu; // return number of variables added to the model
    return status;
}

int SolverBilevel::addDualGammaVariables(int& size)
{
    // Create columns related to the gamma variables
    int sizeGamma = myData->nbTrainSamples;
    char **namesGamma = new char *[sizeGamma];
    double *lbGamma = new double[sizeGamma];
    double *ubGamma = new double[sizeGamma];
	for (int i = 0; i < myData->nbTrainSamples; i++)
	{
        lbGamma[i] = -CPX_INFBOUND;
		ubGamma[i] = 0.0;
        namesGamma[i] = new char[100];
		sprintf(namesGamma[i], "gamma_%d", i);
	}
    int status = solverAddCols(sizeGamma, NULL, lbGamma, ubGamma, NULL, namesGamma);
    
    for (int i = 0; i < sizeGamma; i++)
		delete[] namesGamma[i];
    delete[] namesGamma;
    delete[] lbGamma;
    delete[] ubGamma;

    if (status) return status;

    size = sizeGamma; // return number of variables added to the model
    return status;
}

int SolverBilevel::addFollowerOptConstr()
{
    int nzcnt = myData->nbTrainSamples*4; 
    double rhs = 0;
    char sense = 'L';
    int beg = 0;
    int *rmatind = new int[nzcnt];
    double *rmatval = new double[nzcnt];
    char *rowname = new char[100];
    sprintf(rowname, "follower_opt");
    for (int i=0; i<myData->nbTrainSamples; i++) 
    {
        // variable u
        rmatind[i] = i;
        rmatval[i] = (myData->backOrderCost / (double) myData->nbTrainSamples);
        
        // variable o
        rmatind[i+myData->nbTrainSamples] = myData->nbSamples + i;
        rmatval[i+myData->nbTrainSamples] = (myData->holdingCost / (double) myData->nbTrainSamples);

        // variable mu
        rmatind[i+2*myData->nbTrainSamples] = 2*myData->nbSamples + myData->nbFeatures + i;
        rmatval[i+2*myData->nbTrainSamples] = myData->demand[i];

        // variable gamma
        rmatind[i+3*myData->nbTrainSamples] = 2*myData->nbSamples + myData->nbFeatures + myData->nbTrainSamples + i;
        rmatval[i+3*myData->nbTrainSamples] = - myData->demand[i];
    }
    int status = solverAddRows(1, nzcnt, &rhs, &sense, &beg, rmatind, rmatval, &rowname);

    delete[] rmatind;
    delete[] rmatval;
    delete[] rowname;

    return status;
}

int SolverBilevel::addDualMuConstrs()
{
    // Add constraints related to dual variable mu
    int rcnt = myData->nbTrainSamples; 
    int nzcnt = rcnt; 
    int *rmatbeg = new int[rcnt];
    int *rmatind = new int[nzcnt];
    double *rmatval = new double[nzcnt];
    double *rhs = new double[rcnt];
    char *sense = new char[rcnt];
    char **rowname = new char *[rcnt];
    for (int i=0; i<myData->nbTrainSamples; i++) 
    {
        rmatbeg[i] = i;
        rmatind[i] = 2*myData->nbSamples + myData->nbFeatures + i;
        rmatval[i] = 1; 
        rhs[i] = - (myData->backOrderCost / (double) myData->nbTrainSamples);
        sense[i] = 'G';
        rowname[i] = new char[100];
        sprintf(rowname[i], "dual_mu(%d)", i);
    }
    int status = solverAddRows(rcnt, nzcnt, rhs, sense, rmatbeg, rmatind, rmatval, rowname);

    // Free memory
    delete[] rmatbeg;
    delete[] rmatind;
    delete[] rmatval;
    delete[] rhs;
    delete[] sense;
    for (int i = 0; i < rcnt; i++)
		delete[] rowname[i];
    delete[] rowname;

    return status;
}

int SolverBilevel::addDualGammaConstrs()
{
    int rcnt = myData->nbTrainSamples; 
    int nzcnt = rcnt; 
    int *rmatbeg = new int[rcnt];
    int *rmatind = new int[nzcnt];
    double *rmatval = new double[nzcnt];
    double *rhs = new double[rcnt];
    char *sense = new char[rcnt];
    char **rowname = new char *[rcnt];
    for (int i=0; i<myData->nbTrainSamples; i++) 
    {
        rmatbeg[i] = i;
        rmatind[i] = 2*myData->nbSamples + myData->nbFeatures + myData->nbTrainSamples + i;
        rmatval[i] = 1; 
        rhs[i] = - (myData->holdingCost / (double) myData->nbTrainSamples);
        sense[i] = 'G';
        rowname[i] = new char[100];
        sprintf(rowname[i], "dual_gamma(%d)", i);
    }
    int status = solverAddRows(rcnt, nzcnt, rhs, sense, rmatbeg, rmatind, rmatval, rowname);

    // Free memory
    delete[] rmatbeg;
    delete[] rmatind;
    delete[] rmatval;
    delete[] rhs;
    delete[] sense;
    for (int i = 0; i < rcnt; i++)
		delete[] rowname[i];
    delete[] rowname;

    return status;
}

int SolverBilevel::addDualIndConstrs()
{
    int status = 0;
    int nzcnt = 2*myData->nbTrainSamples;
    int indvar;
    char *indname_str;
    for (int j=0; j<myData->nbFeatures; j++)
	{
        int *linind = new int[nzcnt];
        double *linval = new double[nzcnt];
        for (int i=0; i<myData->nbTrainSamples; i++)
        {
            linind[i] = 2*myData->nbSamples + myData->nbFeatures + i; // mu
            linind[i+myData->nbTrainSamples] = 2*myData->nbSamples + myData->nbFeatures + myData->nbTrainSamples + i; // gamma

            linval[i] = myData->feature_data[i][j];
            linval[i+myData->nbTrainSamples] = - myData->feature_data[i][j];
        }
        indvar = 2*myData->nbSamples + myData->nbFeatures + 2*myData->nbTrainSamples + j;
        indname_str = new char[100];
        sprintf(indname_str, "indDual(%d)", j);
        status = solverAddIndConstr(indvar, false, nzcnt, 0, 'E', linind, linval, indname_str);
        if (status) return status;
        delete[] indname_str;
        delete[] linind;
        delete[] linval;
	}
    return status;
}

int SolverBilevel::solve()
{
    bool debug = true;
    int status = 0;

    std::cout << "Solving optimization problem" << std::endl;

    status = initializeModel(1, "Newsvendor-Features");
    if (status) {
        std::cerr << "Failed to initialize model"<< std::endl;
        return status;
    }

    // ----------------------------------------------------------------------------
    // Add variables to the model

    int sizeU, sizeO, sizeBeta, sizeMu, sizeGamma;

    // Create columns related to the u variables
    if ((status = addUnderageVariables(sizeU))) return status;

    // Create columns related to the o variables
    if ((status = addOverageVariables(sizeO))) return status;

    // Create columns related to the beta variables
    if ((status = addBetaVariables(sizeBeta))) return status;

    // Create columns related to the mu variables
    if ((status = addDualMuVariables(sizeMu))) return status;

    // Create columns related to the gamma variables
    if ((status = addDualGammaVariables(sizeGamma))) return status;

    // Create columns related to the z variables
    int sizeZ = myData->nbFeatures;
    double *lbZ = new double[sizeZ];
    double *ubZ = new double[sizeZ];
    char *xctypeZ = new char[sizeZ];
    char **namesZ = new char *[sizeZ];
	for (int j = 0; j < myData->nbFeatures; j++)
	{
        if (j == 0) {
            lbZ[j] = 1.0;
        } else {
            lbZ[j] = 0.0;
        }
        ubZ[j] = 1.0;
        xctypeZ[j] = 'B';
        namesZ[j] = new char[100];
		sprintf(namesZ[j], "z_%d", j);
	}
    status = solverAddCols(sizeZ, NULL, lbZ, ubZ, xctypeZ, namesZ);
    
    delete[] lbZ;
    delete[] ubZ;
    delete[] xctypeZ;
    for (int j = 0; j < sizeZ; j++)
		delete[] namesZ[j];
    delete[] namesZ;
    if (status) return status;
    
    int sizeVars = sizeU + sizeO + sizeBeta + sizeMu + sizeGamma + sizeZ;

    if (debug)
        std::cout << "Added all columns - OK" << endl;

    // ----------------------------------------------------------------------------
    // Add constraints to the model

    // Add constraints related to underage costs
    if ((status = addUnderageConstrs())) return status;

    // Add constraints related to overage costs
    if ((status = addOverageConstrs())) return status;

    // Add indicator constraints related to the beta variables
    if ((status = addBetaIndConstrs(sizeU+sizeO, sizeU+sizeO+sizeBeta+sizeMu+sizeGamma))) return status;

    // Add constraint related to the lower-level problem's optimality condition
    if ((status = addFollowerOptConstr())) return status;

    // Add constraints related to dual variable mu
    if ((status = addDualMuConstrs())) return status;

    // Add constraints related to dual variable gamma
    if ((status = addDualGammaConstrs())) return status;

    // Add indicator constraints related to the dual variables mu and gamma
    if ((status = addDualIndConstrs())) return status;

    int sizeConstrs = 2*myData->nbSamples + 2*myData->nbTrainSamples + 2*myData->nbFeatures + 1;

    if (debug)
        std::cout << "Added all constraints - OK" << endl;

    // ----------------------------------------------------------------------------
    // Write problem to file
    if (myData->save_model)
    {
        string ext = ".lp";
        string model_path = mySolution->getOutputFilename(ext);
        status = solverWriteModel(model_path.c_str()); // Exporting the model
        if (status) return status;
    }

    // Solve MIP
    status = solverOptimize(MIP_PROBLEM); // Solving the model
    if (status) return status;

    // Get the size of the model
    int numcols, numlinconstrs, numindconstrs, numrows;
    status = solverGetModelDimensions(numcols, numlinconstrs, numindconstrs, numrows);
    if (status) return status;

    std::cout << "Number of columns: " << numcols << std::endl;
    std::cout << "Number of rows: " << numrows << std::endl;
    std::cout << "Number of linear constraints: " << numlinconstrs << std::endl;
    std::cout << "Number of indicator constraints: " << numindconstrs << std::endl;

    if ((numcols != sizeVars) || (numrows != sizeConstrs))
    {
        std::cerr << "ERROR: There is something wrong with the rows or columns of the model." << std::endl;
        quit_solver();
        return 1;
    }

    // Retrieve solution
    int lpstat;
    double objval, bestobjval, mipgap, nodecount;
    double *solution = new double[sizeVars];
    status = solverRetrieveSolution(&lpstat, &objval, &bestobjval, &mipgap, &nodecount, solution, sizeVars);
    if (status) {
        delete[] solution;
        return status;
    }

    // Convert CPLEX status to string 
    string stat_str = getSolverStatusString(lpstat);

    // Print solution array
    std::cout << std::setw(10) << "beta" << std::setw(5) << "z" << endl; 
    int numSelFeats = 0;
    for (int j=0; j<myData->nbFeatures; j++)
    {
        numSelFeats += solution[sizeU+sizeO+sizeBeta+sizeMu+sizeGamma+j];
        std::cout << std::setw(10) << solution[sizeU+sizeO+j] << std::setw(5) << solution[sizeU+sizeO+sizeBeta+sizeMu+sizeGamma+j] << endl;
    }
    std::cout << "Number of selected features: " << numSelFeats << std::endl;

    // Create new solution structure
    this->mySolution->update(numcols, numrows, lpstat, stat_str, objval, bestobjval, nodecount, mipgap, solution);
    delete[] solution;

    status = quit_solver();

    return status;
}

// ***************************************************************************************** // 

int SolverBilevelShuffleSplit::addUnderageVariables(int& size)
{
    // Create columns related to the u variables
    int sizeU = myData->splitSize * myData->nbFolds;
	double *costU = new double[sizeU];
	char **namesU = new char *[sizeU];

    int r = 0;
    for (int k = 0; k < myData->nbFolds; k++)
    {
	    for (int i = 0; i < myData->nbSamples; i++)
        {
            if (myData->crossValSplit[k][i] == 0) // 0 if sample is not selected for the current split
                continue;
            else if (myData->crossValSplit[k][i] == 1) // 1 if sample is in the training set
                costU[r] = 0;
            else if (myData->crossValSplit[k][i] == 2) // 2 if sample is in the validation set
                costU[r] = myData->backOrderCost / ( (double) (myData->nbFolds * myData->splitValSize) );
            else
                return 1;

            namesU[r] = new char[100];
            sprintf(namesU[r], "u_(%d,%d)", k, i);
            r++;
        }
    }
	int status = solverAddCols(sizeU, costU, NULL, NULL, NULL, namesU);
    
    delete[] costU;
    for (int r = 0; r < sizeU; r++)
		delete[] namesU[r];
    delete[] namesU;

    size = sizeU;
    return status;
}
	
int SolverBilevelShuffleSplit::addOverageVariables(int& size)
{
    // Create columns related to the o variables
    int sizeO = myData->splitSize * myData->nbFolds;
    double *costO = new double[sizeO];
	char **namesO = new char *[sizeO];

    int r = 0;
    for (int k = 0; k < myData->nbFolds; k++)
    {
        for (int i = 0; i < myData->nbSamples; i++)
        {
            if (myData->crossValSplit[k][i] == 0) // 0 if sample is not selected for the current split
                continue;
            else if (myData->crossValSplit[k][i] == 1) // 1 if sample is in the training set
                costO[r] = 0;
            else if (myData->crossValSplit[k][i] == 2) // 2 if sample is in the validation set
                costO[r] = myData->holdingCost / ( (double) (myData->nbFolds * myData->splitValSize) );
            else
                return 1;

            namesO[r] = new char[100];
            sprintf(namesO[r], "o_(%d,%d)", k, i);
            r++;
        }
    }
	int status = solverAddCols(sizeO, costO, NULL, NULL, NULL, namesO);

    delete[] costO;
    for (int r = 0; r < sizeO; r++)
		delete[] namesO[r];
    delete[] namesO;

    size = sizeO;
    return status;
}

int SolverBilevelShuffleSplit::addBetaVariables(int& size)
{
    // Create columns related to the beta variables
    int sizeBeta = myData->nbFeatures * myData->nbFolds;
    double *lbBeta = new double[sizeBeta];
    double *ubBeta = new double[sizeBeta];
	char **namesBeta = new char *[sizeBeta];
    int r = 0;
    for (int k = 0; k < myData->nbFolds; k++) {
        for (int j = 0; j < myData->nbFeatures; j++) {
            if (myData->activeFeatures[k][j] == 1) { // feature is active
                lbBeta[r] = -CPX_INFBOUND;
                ubBeta[r] =  CPX_INFBOUND;
            } else { // feature is deactivated
                lbBeta[r] = 0;
                ubBeta[r] = 0;
            }
            namesBeta[r] = new char[100];
            sprintf(namesBeta[r], "beta_(%d,%d)", k, j);
            r++;
        }
    }
	int status = solverAddCols(sizeBeta, NULL, lbBeta, ubBeta, NULL, namesBeta);

    delete[] lbBeta;
    delete[] ubBeta;
    for (int r = 0; r < sizeBeta; r++)
		delete[] namesBeta[r];
    delete[] namesBeta;

    size = sizeBeta;
    return status;
}

int SolverBilevelShuffleSplit::addDualMuVariables(int& size)
{
    // Create columns related to the mu variables
    int sizeMu = myData->nbFolds * myData->splitTrainSize;
    char **namesMu = new char *[sizeMu];
    double *lbMu = new double[sizeMu];
    double *ubMu = new double[sizeMu];
    int r = 0;
    for (int k = 0; k < myData->nbFolds; k++) {
        for (int i = 0; i < myData->nbSamples; i++) {
            if (myData->crossValSplit[k][i] == 1) { // 1 if sample is in the training set
                lbMu[r] = -CPX_INFBOUND;
                ubMu[r] = 0.0;
                namesMu[r] = new char[100];
                sprintf(namesMu[r], "mu_(%d,%d)", k, i);
                r++;
            }
        }
    }
	int status = solverAddCols(sizeMu, NULL, lbMu, ubMu, NULL, namesMu);

    for (int r = 0; r < sizeMu; r++)
		delete[] namesMu[r];
    delete[] namesMu;
    delete[] lbMu;
    delete[] ubMu;

    size = sizeMu;
    return status;
}

int SolverBilevelShuffleSplit::addDualGammaVariables(int& size)
{
    // Create columns related to the gamma variables
    int sizeGamma = myData->nbFolds * myData->splitTrainSize;
    char **namesGamma = new char *[sizeGamma];
    double *lbGamma = new double[sizeGamma];
	double *ubGamma = new double[sizeGamma];

    int r = 0;
    for (int k = 0; k < myData->nbFolds; k++)
    {
        for (int i = 0; i < myData->nbSamples; i++)
        {
            if (myData->crossValSplit[k][i] == 1) // 1 if sample is in the training set
            {
                lbGamma[r] = -CPX_INFBOUND;
                ubGamma[r] = 0.0;
                namesGamma[r] = new char[100];
                sprintf(namesGamma[r], "gamma_(%d,%d)", k, i);
                r++;
            }
        }
    }
	int status = solverAddCols(sizeGamma, NULL, lbGamma, ubGamma, NULL, namesGamma);

    for (int r = 0; r < sizeGamma; r++)
		delete[] namesGamma[r];
    delete[] namesGamma;
    delete[] lbGamma;
    delete[] ubGamma;

    size = sizeGamma;
    return status;
}

int SolverBilevelShuffleSplit::addZVariables(int& size)
{
    // Create columns related to the z variables
    int sizeZ = myData->nbFeatures;
    char *xctypeZ = new char[sizeZ];
    char **namesZ = new char *[sizeZ];
    double *lbZ = new double[sizeZ];
	for (int j = 0; j < myData->nbFeatures; j++)
	{
        if (j == 0) {
            lbZ[j] = 1.0;
        } else {
            lbZ[j] = 0.0;
        }
        xctypeZ[j] = 'B';
        namesZ[j] = new char[100];
		sprintf(namesZ[j], "z_%d", j);
	}
	int status = solverAddCols(sizeZ, NULL, lbZ, NULL, xctypeZ, namesZ);
    
    delete[] xctypeZ;
    for (int j = 0; j < sizeZ; j++)
		delete[] namesZ[j];
    delete[] namesZ;
    delete[] lbZ;

    size = sizeZ;
    return status;
}

int SolverBilevelShuffleSplit::addUnderageConstrs()
{
    // Add constraints related to underage costs
    int rcnt = myData->splitSize * myData->nbFolds; // An integer that indicates the number of new rows to be added to the constraint matrix.
    int ncolsperrow = myData->nbFeatures + 1;
    int nzcnt = rcnt * ncolsperrow; // An integer that indicates the number of nonzero constraint coefficients to be added to the constraint matrix. This specifies the length of the arrays rmatind and rmatval.
    int *rmatbeg = new int[rcnt];
    int *rmatind = new int[nzcnt];
    double *rmatval = new double[nzcnt];
    double *rhs = new double[rcnt];
    char *sense = new char[rcnt];
    char **rowname = new char *[rcnt];
    int r = 0;
    for (int k=0; k<myData->nbFolds; k++)
    {
        for (int i=0; i<myData->nbSamples; i++) 
        {
            if (myData->crossValSplit[k][i] == 0) // 0 if sample is not selected for the current split
                continue;

            rmatbeg[r] = r*ncolsperrow;
            rmatind[r*ncolsperrow] = r; // variable u_{ik}
            rmatval[r*ncolsperrow] = 1; 
            for (int j=0; j<myData->nbFeatures; j++)
            {
                rmatind[r*ncolsperrow+j+1] = 2*myData->splitSize*myData->nbFolds + (k * myData->nbFeatures) + j; // variable beta^j_k
                rmatval[r*ncolsperrow+j+1] = myData->feature_data[i][j]; // feature x^j_i
            }
            rhs[r] = myData->demand[i]; // demand d_i
            sense[r] = 'G';
            rowname[r] = new char[100];
            sprintf(rowname[r], "underage(%d,%d)", k, i);
            r++;
        }
    }
    int status = solverAddRows(rcnt, nzcnt, rhs, sense, rmatbeg, rmatind, rmatval, rowname);

    // Free memory
    delete[] rmatbeg;
    delete[] rmatind;
    delete[] rmatval;
    delete[] rhs;
    delete[] sense;
    for (int r = 0; r < rcnt; r++)
		delete[] rowname[r];
    delete[] rowname;

    return status;
}

int SolverBilevelShuffleSplit::addOverageConstrs()
{
    // Add constraints related to overage costs
    int rcnt = myData->splitSize * myData->nbFolds; // An integer that indicates the number of new rows to be added to the constraint matrix.
    int ncolsperrow = myData->nbFeatures + 1;
    int nzcnt = rcnt * ncolsperrow; // An integer that indicates the number of nonzero constraint coefficients to be added to the constraint matrix. This specifies the length of the arrays rmatind and rmatval.
    int *rmatbeg = new int[rcnt];
    int *rmatind = new int[nzcnt];
    double *rmatval = new double[nzcnt];
    double *rhs = new double[rcnt];
    char *sense = new char[rcnt];
    char **rowname = new char *[rcnt];
    int r = 0;
    for (int k=0; k<myData->nbFolds; k++)
    {
        for (int i=0; i<myData->nbSamples; i++) 
        {
            if (myData->crossValSplit[k][i] == 0) // 0 if sample is not selected for the current split
                continue;

            rmatbeg[r] = r*ncolsperrow;
            rmatind[r*ncolsperrow] = myData->splitSize*myData->nbFolds + r; // variable o
            rmatval[r*ncolsperrow] = 1; 
            for (int j=0; j<myData->nbFeatures; j++)
            {
                rmatind[r*ncolsperrow+j+1] = 2*myData->splitSize*myData->nbFolds + (k * myData->nbFeatures) + j; // variable beta
                rmatval[r*ncolsperrow+j+1] = - myData->feature_data[i][j];
            }
            rhs[r] = - myData->demand[i];
            sense[r] = 'G';
            rowname[r] = new char[100];
            sprintf(rowname[r], "overage(%d,%d)", k, i);
            r++;
        }
    }
    int status = solverAddRows(rcnt, nzcnt, rhs, sense, rmatbeg, rmatind, rmatval, rowname);

    // Free memory
    delete[] rmatbeg;
    delete[] rmatind;
    delete[] rmatval;
    delete[] rhs;
    delete[] sense;
    for (int r = 0; r < rcnt; r++)
		delete[] rowname[r];
    delete[] rowname;

    return status;
}

int SolverBilevelShuffleSplit::addBetaIndConstrs(const int startBeta, const int startZ)
{
    // Add indicator constraints related to the beta and z variables
    int status = 0;
    int indvar, ind;
    double val;
    char *indname_str;
    for (int k=0; k<myData->nbFolds; k++)
    {
        for (int j=0; j<myData->nbFeatures; j++)
        {
            if (myData->activeFeatures[k][j] == 1) // feature is active
            {
                indvar = startZ + j;
                ind = startBeta + (k * myData->nbFeatures) + j;
                val = 1;
                indname_str = new char[100];
                sprintf(indname_str, "indBeta(%d,%d)", k, j);
                status = solverAddIndConstr(indvar, true, 1, 0, 'E', &ind, &val, indname_str);
                delete[] indname_str;
                if (status) return status;
            }
        }
    }
    return status;
}

int SolverBilevelShuffleSplit::addFollowerOptConstr()
{
    // Add constraint related to the lower-level problem's optimality condition    
    int rcnt = myData->nbFolds; // An integer that indicates the number of new rows to be added to the constraint matrix.
    int nzcnt = rcnt * (myData->splitTrainSize * 4);
    int *rmatbeg = new int[rcnt];
    int *rmatind = new int[nzcnt];
    double *rmatval = new double[nzcnt];
    double *rhs = new double[rcnt];
    char *sense = new char[rcnt];
    char **rowname = new char *[rcnt];

    int p_idx = 0; // primal index
    int d_idx = 0; // dual index
    int r = 0;
    for (int k=0; k<myData->nbFolds; k++)
    {
        rmatbeg[k] = r;
        for (int i=0; i<myData->nbSamples; i++) 
        {            
            if (myData->crossValSplit[k][i] == 0) // 0 if sample is not selected for the current split
                continue;

            if (myData->crossValSplit[k][i] == 2) // 2 if sample is in the validation set
            {
                p_idx++; continue;
            }

            // Primal variables
            // variable u
            rmatind[r] = p_idx;
            rmatval[r] = (myData->backOrderCost / ((double) myData->splitTrainSize));
            
            // variable o
            rmatind[r+1] = myData->splitSize*myData->nbFolds + p_idx;
            rmatval[r+1] = (myData->holdingCost / ((double) myData->splitTrainSize));

            // Dual variables
            // variable mu
            rmatind[r+2] = 2*myData->splitSize*myData->nbFolds + myData->nbFeatures*myData->nbFolds + d_idx;
            rmatval[r+2] = myData->demand[i];

            // variable gamma
            rmatind[r+3] = 2*myData->splitSize*myData->nbFolds + myData->nbFeatures*myData->nbFolds + myData->nbFolds*myData->splitTrainSize + d_idx;
            rmatval[r+3] = - myData->demand[i];

            p_idx++;
            d_idx++;
            r += 4;
        }
        rhs[k] = 0;
        sense[k] = 'L';
        rowname[k] = new char[100];
        sprintf(rowname[k], "follower_opt(%d)", k);
    }
    int status = solverAddRows(rcnt, nzcnt, rhs, sense, rmatbeg, rmatind, rmatval, rowname);

    // Free memory
    delete[] rmatbeg;
    delete[] rmatind;
    delete[] rmatval;
    delete[] rhs;
    delete[] sense;
    for (int r = 0; r < rcnt; r++)
		delete[] rowname[r];
    delete[] rowname;

    return status;
}


int SolverBilevelShuffleSplit::addDualMuConstrs(const int startMu)
{
    // Add constraints related to dual variable mu
    int rcnt, nzcnt;
    rcnt = nzcnt = myData->nbFolds * myData->splitTrainSize; 
    
    int *rmatbeg = new int[rcnt];
    int *rmatind = new int[nzcnt];
    double *rmatval = new double[nzcnt];
    double *rhs = new double[rcnt];
    char *sense = new char[rcnt];
    char **rowname = new char *[rcnt];
    int r = 0;
    for (int k=0; k<myData->nbFolds; k++)
    {
        for (int i=0; i<myData->nbSamples; i++) 
        {
            if (myData->crossValSplit[k][i] == 1) // 1 if sample is in the training set  
            {
                rmatbeg[r] = r;
                rmatind[r] = startMu + r;
                rmatval[r] = 1; 
                rhs[r] = - (myData->backOrderCost / ((double) myData->splitTrainSize));
                sense[r] = 'G';
                rowname[r] = new char[100];
                sprintf(rowname[r], "dual_mu(%d,%d)", k, i);
                r++;
            }         
        }
    }
    int status = solverAddRows(rcnt, nzcnt, rhs, sense, rmatbeg, rmatind, rmatval, rowname);

    // Free memory
    delete[] rmatbeg;
    delete[] rmatind;
    delete[] rmatval;
    delete[] rhs;
    delete[] sense;
    for (int i = 0; i < rcnt; i++)
		delete[] rowname[i];
    delete[] rowname;

    return status;
}

int SolverBilevelShuffleSplit::addDualGammaConstrs(const int startGamma)
{
    // Add constraints related to dual variable gamma
    int rcnt, nzcnt;
    rcnt = nzcnt = myData->nbFolds * myData->splitTrainSize;
    int *rmatbeg = new int[rcnt];
    int *rmatind = new int[nzcnt];
    double *rmatval = new double[nzcnt];
    double *rhs = new double[rcnt];
    char *sense = new char[rcnt];
    char **rowname = new char *[rcnt];
    int r = 0;
    for (int k=0; k<myData->nbFolds; k++)
    {
        for (int i=0; i<myData->nbSamples; i++) 
        {
            if (myData->crossValSplit[k][i] == 1) // 1 if sample is in the training set  
            {
                rmatbeg[r] = r;
                rmatind[r] = startGamma + r;
                rmatval[r] = 1; 
                rhs[r] = - (myData->holdingCost / ((double) myData->splitTrainSize));
                sense[r] = 'G';
                rowname[r] = new char[100];
                sprintf(rowname[r], "dual_gamma(%d,%d)", k, i);
                r++;
            }
        }
    }
    int status = solverAddRows(rcnt, nzcnt, rhs, sense, rmatbeg, rmatind, rmatval, rowname);

    // Free memory
    delete[] rmatbeg;
    delete[] rmatind;
    delete[] rmatval;
    delete[] rhs;
    delete[] sense;
    for (int i = 0; i < rcnt; i++)
		delete[] rowname[i];
    delete[] rowname;

    return status;
}

int SolverBilevelShuffleSplit::addDualIndConstrs()
{
    // Add indicator constraints related to the dual variables mu and gamma
    int status = 0;
    int nzcnt, indvar, d_idx;
    char *indname_str;
    for (int j=0; j<myData->nbFeatures; j++)
    {
        d_idx = 0;
        for (int k=0; k<myData->nbFolds; k++)
        {   
            nzcnt = 2*myData->splitTrainSize;
            int *linind = new int[nzcnt];
            double *linval = new double[nzcnt];
            int r = 0;
            for (int i=0; i<myData->nbSamples; i++)
            {
                if (myData->crossValSplit[k][i] == 1) // 1 if sample is in the training set  
                {
                    linind[r] = 2*myData->splitSize*myData->nbFolds + myData->nbFeatures*myData->nbFolds + d_idx; // mu
                    linind[r+1] = 2*myData->splitSize*myData->nbFolds + myData->nbFeatures*myData->nbFolds + myData->nbFolds*myData->splitTrainSize + d_idx; // gamma

                    linval[r] = myData->feature_data[i][j];
                    linval[r+1] = - myData->feature_data[i][j];
                    r += 2;
                    d_idx++;
                }
            }
            indvar = 2*myData->splitSize*myData->nbFolds + myData->nbFeatures*myData->nbFolds + 2*myData->nbFolds*myData->splitTrainSize + j; // z
            indname_str = new char[100];
            sprintf(indname_str, "indDual(%d,%d)", k, j);
            
            if (myData->activeFeatures[k][j] == 1) // only include this contraint if feature j is active in fold k
            {
                status = solverAddIndConstr(indvar, false, nzcnt, 0, 'E', linind, linval, indname_str);
                if (status) return status;
            }
            delete[] indname_str;
            delete[] linind;
            delete[] linval;
        }
    }
    return status;
}

int SolverBilevelShuffleSplit::solve()
{
    bool debug = true;
    int status = 0;

    // Build optimization model
    std::cout << "Solving optimization problem" << endl;

    status = initializeModel(1, "Newsvendor-Features");
    if (status) {
        std::cerr << "Failed to initialize model" << std::endl;
        return status;
    }

    // ----------------------------------------------------------------------------
    // Add variables to the model
    int sizeU, sizeO, sizeBeta, sizeMu, sizeGamma, sizeZ;

    // Create columns related to the u variables
    if ((status = addUnderageVariables(sizeU))) return status;

    // Create columns related to the o variables
    if ((status = addOverageVariables(sizeO))) return status;

    // Create columns related to the beta variables
    if ((status = addBetaVariables(sizeBeta))) return status;

    // Create columns related to the mu variables
    if ((status = addDualMuVariables(sizeMu))) return status;

    // Create columns related to the gamma variables
    if ((status = addDualGammaVariables(sizeGamma))) return status;

    // Create columns related to the z variables
    if ((status = addZVariables(sizeZ))) return status;

    int sizeVars = sizeU + sizeO + sizeBeta + sizeMu + sizeGamma + sizeZ;

    if (debug)
        std::cout << "Added all columns - OK" << endl;

    // ----------------------------------------------------------------------------
    // Add constraints to the model
    
    // Add constraints related to underage costs
    if ((status = addUnderageConstrs())) return status;

    // Add constraints related to overage costs
    if ((status = addOverageConstrs())) return status;

    // Add indicator constraints related to the beta and z variables
    if ((status = addBetaIndConstrs(sizeU+sizeO, sizeU+sizeO+sizeBeta+sizeMu+sizeGamma))) return status;

    // Add constraint related to the lower-level problem's optimality condition
    if ((status = addFollowerOptConstr())) return status;

    // Add constraints related to dual variable mu
    if ((status = addDualMuConstrs(sizeU+sizeO+sizeBeta))) return status;

    // Add constraints related to dual variable gamma
    if ((status = addDualGammaConstrs(sizeU+sizeO+sizeBeta+sizeMu))) return status;

    // Add indicator constraints related to the dual variables mu and gamma
    if ((status = addDualIndConstrs())) return status;

    int sizeConstrs = 2*myData->splitSize*myData->nbFolds + myData->nbFolds + 2*myData->splitTrainSize*myData->nbFolds + 2*myData->nbFeatures*myData->nbFolds;

    if (debug)
        std::cout << "Added all constraints - OK" << endl;

    // ----------------------------------------------------------------------------
    // Write problem to file
    if (myData->save_model)
    {
        string ext = ".lp";
        string model_path = mySolution->getOutputFilename(ext);
        status = solverWriteModel(model_path.c_str()); // Exporting the model
        if (status) return status;
    }

    // Solve MIP
    status = solverOptimize(MIP_PROBLEM); // Solving the model
    if (status) return status;
    
    // Get the size of the model
    int numcols, numlinconstrs, numindconstrs, numrows;
    status = solverGetModelDimensions(numcols, numlinconstrs, numindconstrs, numrows);
    if (status) return status;

    std::cout << "Number of columns: " << numcols << std::endl;
    std::cout << "Number of rows: " << numrows << std::endl;
    std::cout << "Number of linear constraints: " << numlinconstrs << std::endl;
    std::cout << "Number of indicator constraints: " << numindconstrs << std::endl;

    if ((numcols != sizeVars) || (numrows != sizeConstrs))
    {
        std::cerr << "ERROR: There is something wrong with the model dimensions." << std::endl;
        quit_solver();
        return 1;
    }

    // Retrieve solution
    int lpstat;
    double objval, bestobjval, mipgap, nodecount;
    double *solution = new double[sizeVars];
    status = solverRetrieveSolution(&lpstat, &objval, &bestobjval, &mipgap, &nodecount, solution, sizeVars);
    if (status) {
        delete[] solution;
        return status;
    }

    // Convert status to string 
    string stat_str = getSolverStatusString(lpstat);
    
    // Print solution
    std::cout << std::setw(5) << "z" << "\t";
    for (int k=0; k<myData->nbFolds; k++)
    {
        std::cout << std::setw(9) << "beta_" << k;
    }
    std::cout << endl;

    int numSelFeats = 0;
    for (int j=0; j<myData->nbFeatures; j++)
    {
        std::cout << std::setw(5) << solution[sizeU+sizeO+sizeBeta+sizeMu+sizeGamma+j] << "\t";

        for (int k=0; k<myData->nbFolds; k++)
        {
            std::cout << std::setw(10) << solution[sizeU+sizeO+k*(myData->nbFeatures)+j];
        }
        std::cout << endl;

        numSelFeats += solution[sizeU+sizeO+sizeBeta+sizeMu+sizeGamma+j];
    }
    std::cout << "Number of selected features: " << numSelFeats << std::endl;

    // Create new solution structure
    this->mySolution->update(numcols, numrows, lpstat, stat_str, objval, bestobjval, nodecount, mipgap, solution);
    delete[] solution;

    status = quit_solver();

    return status;
}

