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
        fprintf(stderr, "Failed to create solver structure: no solver matching the input could be found \n");
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
        fprintf(stderr, "Failed to create solver structure for subproblem: no solver matching the input could be found \n");
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

int Solver::quit_solver(void *env, void *model)
{
    int status = 0;
    char errbuf[CPXMESSAGEBUFSIZE];

    CPXLPptr lp = (CPXLPptr) model;

    // Free the problem as allocated by CPXcreateprob and CPXreadcopyprob, if necessary
    if (lp != NULL) {
        int xstatus = CPXfreeprob((CPXENVptr) env, &lp);
        if (!status) status = xstatus;
        if (status) {
            fprintf(stderr, "Failed to free memory for problem: %s\n", 
                    CPXgeterrorstring((CPXENVptr) env, status, errbuf));
        }
    }

    // Free the CPLEX environment, if necessary
    if (env != NULL) {
        int xstatus = CPXcloseCPLEX((CPXENVptr *) &env);
        if (!status) status = xstatus;
        if (status) {
            fprintf(stderr, "Failed to close CPLEX: %s\n", 
                    CPXgeterrorstring((CPXENVptr) env, status, errbuf));
        }
    }
    return status;
}

int Solver::initializeModel(int logToConsole, char const *modelname)
{
    int status = 0;

    this->env = CPXopenCPLEX(&status); // Open CPLEX environment
    if (status) return status;

    CPXsetintparam((CPXENVptr) this->env, CPXPARAM_ScreenOutput, logToConsole); // Switching ON the display
    CPXsetintparam((CPXENVptr) this->env, CPXPARAM_Read_DataCheck, CPX_DATACHECK_WARN); // Print warnings
    CPXsetintparam((CPXENVptr) this->env, CPX_PARAM_THREADS, myData->nbThreads);		  // number of threads
    if (myData->timeLimit > 0)
        CPXsetdblparam((CPXENVptr) this->env, CPX_PARAM_TILIM, myData->timeLimit); // sets time limit for the solver.

    this->model = CPXcreateprob((CPXENVptr) this->env, &status, modelname); // Create LP problem as a container
    return status;
}

int Solver::solverAddCols(void *env, void *model, int ccnt, double *obj, double *lb, double *ub, char *vtype, char **colname)
{
    int status = 0;

    char errbuf[CPXMESSAGEBUFSIZE];
    status = CPXnewcols((CPXENVptr) env, (CPXLPptr) model, ccnt, obj, lb, ub, vtype, colname);
    if (status) fprintf(stderr, "Failed to add variables: %s\n", CPXgeterrorstring((CPXENVptr) env, status, errbuf));
    return status;
}

int Solver::solverAddRows(void *env, void *model, int rcnt, int nzcnt, double *rhs, char *sense, int *rmatbeg, int *rmatind, double *rmatval, char **rowname)
{
    int status = 0;

    status = CPXaddrows((CPXENVptr) env, (CPXLPptr) model, 0, rcnt, nzcnt, rhs, sense, rmatbeg, rmatind, rmatval, NULL, rowname);
    return status;
}

int Solver::solverAddIndConstr(void *env, void *model, int indvar, int complemented, int nzcnt, double rhs, int sense, int *linind, double *linval, char *indname_str)
{
    int status = 0;

    status = CPXaddindconstr((CPXENVptr) env, (CPXLPptr) model, indvar, complemented, nzcnt, rhs, sense, linind, linval, indname_str);
    return status;
}

int Solver::solverWriteModel(void *env, void *model, char const *filename)
{
    int status = 0;

    status = CPXwriteprob((CPXENVptr) env, (CPXLPptr) model, filename, NULL); // Exporting the model
    return status;
}

int Solver::solverGetModelDimensions(void *env, void *model, int& numcols, int& numlinconstrs, int& numindconstrs, int& numrows)
{
    int status = 0;
    numcols = CPXgetnumcols((CPXENVptr) env, (CPXLPptr) model); // Get number of columns
    numlinconstrs = CPXgetnumrows((CPXENVptr) env, (CPXLPptr) model); // Get number of rows 
    numindconstrs = CPXgetnumindconstrs((CPXENVptr) env, (CPXLPptr) model); // Get number of indicator constraints
    numrows = numlinconstrs + numindconstrs;
    return status;
}

int Solver::solverOptimize(void *env, void *model, ModelType model_type)
{
    int status = 0;

    char errbuf[CPXMESSAGEBUFSIZE];
    if (model_type == LP_PROBLEM) {
        status = CPXlpopt((CPXENVptr) env, (CPXLPptr) model);
    } else if (model_type == MIP_PROBLEM) {
        status = CPXmipopt((CPXENVptr) env, (CPXLPptr) model);
    }
    if (status) fprintf(stderr, "Failed to optimize: %s\n", CPXgeterrorstring((CPXENVptr) env, status, errbuf));
    return status;
}

int Solver::solverRetrieveSolution(void *env, void *model, int *modelstatus, double *objval, double *bestobjval, double *mipgap, double *nodecount, double *solution, int sizeVars)
{
    int status = 0;

    char errbuf[CPXMESSAGEBUFSIZE];
    status = CPXsolution((CPXENVptr) env, (CPXLPptr) model, modelstatus, objval, solution, NULL, NULL, NULL); // Get solution array
    if (status) fprintf(stderr, "Failed to retrieve solution: %s\n", CPXgeterrorstring((CPXENVptr) env, status, errbuf));

    if (nodecount){
        *nodecount = CPXgetnodecnt((CPXENVptr) env, (CPXLPptr) model); // Get node count
    }

    if (mipgap) {
        status = CPXgetmiprelgap((CPXENVptr) env, (CPXLPptr) model, mipgap); // Get MIP gap
        if (status) fprintf(stderr, "Failed to retrieve MIP gap: %s\n", CPXgeterrorstring((CPXENVptr) env, status, errbuf));
    }

    if (bestobjval) {
        status = CPXgetbestobjval((CPXENVptr) env, (CPXLPptr) model, bestobjval); // Get best objective value
        if (status) fprintf(stderr, "Failed to retrieve best obj val: %s\n", CPXgeterrorstring((CPXENVptr) env, status, errbuf));
    }
    return status;
}


int SolverBilevel::addUnderageVariables(void *env, void *model, int& size)
{
    // Create columns related to the u variables
    int sizeU = myData->nbSamples;
	double costU[sizeU];
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
    int status = solverAddCols(env, model, sizeU, costU, NULL, NULL, NULL, namesU);
    for (int i = 0; i < sizeU; i++)
		delete[] namesU[i];
    delete[] namesU;

    size = sizeU; // return number of variables added to the model
    return status;
}

int SolverBilevel::addOverageVariables(void *env, void *model, int& size)
{
    // Create columns related to the o variables
    int sizeO = myData->nbSamples;
    double costO[sizeO];
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
    int status = solverAddCols(env, model, sizeO, costO, NULL, NULL, NULL, namesO);

    for (int i = 0; i < sizeO; i++)
		delete[] namesO[i];
    delete[] namesO;

    size = sizeO; // return number of variables added to the model
    return status;
}

int SolverBilevel::addDualMuVariables(void *env, void *model, int& size)
{
    // Create columns related to the mu variables
    int sizeMu = myData->nbTrainSamples;
	char **namesMu = new char *[sizeMu];
    double lbMu[sizeMu];
	double ubMu[sizeMu];
	for (int i = 0; i < myData->nbTrainSamples; i++)
	{
        lbMu[i] = -CPX_INFBOUND;
		ubMu[i] = 0.0;
        namesMu[i] = new char[100];
		sprintf(namesMu[i], "mu_%d", i);
	}
    int status = solverAddCols(env, model, sizeMu, NULL, lbMu, ubMu, NULL, namesMu);

    for (int i = 0; i < sizeMu; i++)
		delete[] namesMu[i];
    delete[] namesMu;

    size = sizeMu; // return number of variables added to the model
    return status;
}

int SolverBilevel::addDualGammaVariables(void *env, void *model, int& size)
{
    // Create columns related to the gamma variables
    int sizeGamma = myData->nbTrainSamples;
    char **namesGamma = new char *[sizeGamma];
    double lbGamma[sizeGamma];
	double ubGamma[sizeGamma];
	for (int i = 0; i < myData->nbTrainSamples; i++)
	{
        lbGamma[i] = -CPX_INFBOUND;
		ubGamma[i] = 0.0;
        namesGamma[i] = new char[100];
		sprintf(namesGamma[i], "gamma_%d", i);
	}
    int status = solverAddCols(env, model, sizeGamma, NULL, lbGamma, ubGamma, NULL, namesGamma);
    
    for (int i = 0; i < sizeGamma; i++)
		delete[] namesGamma[i];
    delete[] namesGamma;

    if (status) return status;

    size = sizeGamma; // return number of variables added to the model
    return status;
}

int SolverBilevel::addFollowerOptConstr(void *env, void *model)
{
    int nzcnt = myData->nbTrainSamples*4; 
    double rhs = 0;
    char sense = 'L';
    int beg = 0;
    int rmatind[nzcnt];
    double rmatval[nzcnt];
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
    int status = solverAddRows(env, model, 1, nzcnt, &rhs, &sense, &beg, rmatind, rmatval, &rowname);

    delete[] rowname;

    return status;
}

int SolverBilevel::addDualMuConstrs(void *env, void *model)
{
    // Add constraints related to dual variable mu
    int rcnt = myData->nbTrainSamples; 
    int nzcnt = rcnt; 
    int rmatbeg[rcnt];
    int rmatind[nzcnt];
    double rmatval[nzcnt];
    double rhs[rcnt];
    char sense[rcnt];
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
    int status = solverAddRows(env, model, rcnt, nzcnt, rhs, sense, rmatbeg, rmatind, rmatval, rowname);

    for (int i = 0; i < rcnt; i++)
		delete[] rowname[i];
    delete[] rowname;

    return status;
}

int SolverBilevel::addDualGammaConstrs(void *env, void *model)
{
    int rcnt = myData->nbTrainSamples; 
    int nzcnt = rcnt; 
    int rmatbeg[rcnt];
    int rmatind[nzcnt];
    double rmatval[nzcnt];
    double rhs[rcnt];
    char sense[rcnt];
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
    int status = solverAddRows(env, model, rcnt, nzcnt, rhs, sense, rmatbeg, rmatind, rmatval, rowname);

    for (int i = 0; i < rcnt; i++)
		delete[] rowname[i];
    delete[] rowname;

    return status;
}

int SolverBilevel::addDualIndConstrs(void *env, void *model)
{
    int status = 0;
    int nzcnt = 2*myData->nbTrainSamples;
    int indvar;
    char *indname_str;
    for (int j=0; j<myData->nbFeatures; j++)
	{
        int linind[nzcnt];
        double linval[nzcnt];
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
        status = solverAddIndConstr(env, model, indvar, false, nzcnt, 0, 'E', linind, linval, indname_str);
        if (status) return status;
        delete[] indname_str;
	}
    return status;
}

int SolverBilevel::solve()
{
    bool debug = true;
    int status = 0;

    std::cout << "Solving optimization problem" << endl;

    status = initializeModel(1, "Newsvendor-Features");
    if (status) {
        fprintf(stderr, "Failed to initialize model\n");
        return status;
    }

    // ----------------------------------------------------------------------------
    // Add variables to the model

    int sizeU, sizeO, sizeBeta, sizeMu, sizeGamma;

    // Create columns related to the u variables
    if (status = addUnderageVariables(env, model, sizeU)) return status;

    // Create columns related to the o variables
    if (status = addOverageVariables(env, model, sizeO)) return status;

    // Create columns related to the beta variables
    if (status = addBetaVariables(env, model, sizeBeta)) return status;

    // Create columns related to the mu variables
    if (status = addDualMuVariables(env, model, sizeMu)) return status;

    // Create columns related to the gamma variables
    if (status = addDualGammaVariables(env, model, sizeGamma)) return status;

    // Create columns related to the z variables
    int sizeZ = myData->nbFeatures;
    double lbZ[sizeZ];
    double ubZ[sizeZ];
    char xctypeZ[sizeZ];
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

        if ((myData->set_value_z) && (j > 0))
        {
            if (j <= myData->nbInformativeFeatures)
                lbZ[j] = 1.0;
            else
                ubZ[j] = 0.0;
        }
	}
    status = solverAddCols(env, model, sizeZ, NULL, lbZ, ubZ, xctypeZ, namesZ);
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
    if (status = addUnderageConstrs(env, model)) return status;

    // Add constraints related to overage costs
    if (status = addOverageConstrs(env, model)) return status;

    // Add indicator constraints related to the beta variables
    if (status = addBetaIndConstrs(env, model, sizeU+sizeO, sizeU+sizeO+sizeBeta+sizeMu+sizeGamma)) return status;

    // Add constraint related to the lower-level problem's optimality condition
    if (status = addFollowerOptConstr(env, model)) return status;

    // Add constraints related to dual variable mu
    if (status = addDualMuConstrs(env, model)) return status;

    // Add constraints related to dual variable gamma
    if (status = addDualGammaConstrs(env, model)) return status;

    // Add indicator constraints related to the dual variables mu and gamma
    if (status = addDualIndConstrs(env, model)) return status;

    int sizeConstrs = 2*myData->nbSamples + 2*myData->nbTrainSamples + 2*myData->nbFeatures + 1;

    if (debug)
        std::cout << "Added all constraints - OK" << endl;

    // ----------------------------------------------------------------------------
    // Write problem to file
    if (myData->save_model)
    {
        string ext = ".lp";
        string model_path = mySolution->getOutputFilename(ext);
        status = solverWriteModel(env, model, model_path.c_str()); // Exporting the model
        if (status) return status;
    }

    // Solve MIP
    status = solverOptimize(env, model, MIP_PROBLEM); // Solving the model
    if (status) return status;

    // Get the size of the model
    int numcols, numlinconstrs, numindconstrs, numrows;
    status = solverGetModelDimensions(env, model, numcols, numlinconstrs, numindconstrs, numrows);
    if (status) return status;

    printf("Number of columns: %d\n", numcols);
    printf("Number of rows: %d\n", numrows);
    printf("Number of linear constraints: %d\n", numlinconstrs);
    printf("Number of indicator constraints: %d\n", numindconstrs);

    if ((numcols != sizeVars) || (numrows != sizeConstrs))
    {
        fprintf(stderr, "ERROR: There is something wrong with the rows or columns of the model. \n");
        quit_solver(env, model);
        return 1;
    }

    // Retrieve solution
    int lpstat;
    double objval, bestobjval, mipgap, nodecount;
    double solution[sizeVars];
    status = solverRetrieveSolution(env, model, &lpstat, &objval, &bestobjval, &mipgap, &nodecount, solution, sizeVars);
    if (status) return status;

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
    printf("Number of selected features: %d\n", numSelFeats);

    // Create new solution structure
    this->mySolution->update(numcols, numrows, lpstat, stat_str, objval, bestobjval, nodecount, mipgap, solution);

    status = quit_solver(env, model);

    return status;
}

// ***************************************************************************************** // 

int SolverBilevelShuffleSplit::addUnderageVariables(void *env, void *model, int& size)
{
    // Create columns related to the u variables
    int sizeU = myData->splitSize * myData->nbFolds;
	double costU[sizeU];
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
	int status = solverAddCols(env, model, sizeU, costU, NULL, NULL, NULL, namesU);
    
    for (int r = 0; r < sizeU; r++)
		delete[] namesU[r];
    delete[] namesU;

    size = sizeU;
    return status;
}
	
int SolverBilevelShuffleSplit::addOverageVariables(void *env, void *model, int& size)
{
    // Create columns related to the o variables
    int sizeO = myData->splitSize * myData->nbFolds;
    double costO[sizeO];
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
	int status = solverAddCols(env, model, sizeO, costO, NULL, NULL, NULL, namesO);

    for (int r = 0; r < sizeO; r++)
		delete[] namesO[r];
    delete[] namesO;

    size = sizeO;
    return status;
}

int SolverBilevelShuffleSplit::addBetaVariables(void *env, void *model, int& size)
{
    // Create columns related to the beta variables
    int sizeBeta = myData->nbFeatures * myData->nbFolds;
    double lbBeta[sizeBeta];
    double ubBeta[sizeBeta];
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
	int status = solverAddCols(env, model, sizeBeta, NULL, lbBeta, ubBeta, NULL, namesBeta);

    for (int r = 0; r < sizeBeta; r++)
		delete[] namesBeta[r];
    delete[] namesBeta;

    size = sizeBeta;
    return status;
}

int SolverBilevelShuffleSplit::addDualMuVariables(void *env, void *model, int& size)
{
    // Create columns related to the mu variables
    int sizeMu = myData->nbFolds * myData->splitTrainSize;
	char **namesMu = new char *[sizeMu];
    double lbMu[sizeMu];
	double ubMu[sizeMu];
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
	int status = solverAddCols(env, model, sizeMu, NULL, lbMu, ubMu, NULL, namesMu);

    for (int r = 0; r < sizeMu; r++)
		delete[] namesMu[r];
    delete[] namesMu;

    size = sizeMu;
    return status;
}

int SolverBilevelShuffleSplit::addDualGammaVariables(void *env, void *model, int& size)
{
    // Create columns related to the gamma variables
    int sizeGamma = myData->nbFolds * myData->splitTrainSize;
    char **namesGamma = new char *[sizeGamma];
    double lbGamma[sizeGamma];
	double ubGamma[sizeGamma];

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
	int status = solverAddCols(env, model, sizeGamma, NULL, lbGamma, ubGamma, NULL, namesGamma);

    for (int r = 0; r < sizeGamma; r++)
		delete[] namesGamma[r];
    delete[] namesGamma;

    size = sizeGamma;
    return status;
}

int SolverBilevelShuffleSplit::addZVariables(void *env, void *model, int& size)
{
    // Create columns related to the z variables
    int sizeZ = myData->nbFeatures;
    char xctypeZ[sizeZ];
    char **namesZ = new char *[sizeZ];
    double lbZ[sizeZ];
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
	int status = solverAddCols(env, model, sizeZ, NULL, lbZ, NULL, xctypeZ, namesZ);
    
    for (int j = 0; j < sizeZ; j++)
		delete[] namesZ[j];
    delete[] namesZ;

    size = sizeZ;
    return status;
}

int SolverBilevelShuffleSplit::addUnderageConstrs(void *env, void *model)
{
    // Add constraints related to underage costs
    int rcnt = myData->splitSize * myData->nbFolds; // An integer that indicates the number of new rows to be added to the constraint matrix.
    int ncolsperrow = myData->nbFeatures + 1;
    int nzcnt = rcnt * ncolsperrow; // An integer that indicates the number of nonzero constraint coefficients to be added to the constraint matrix. This specifies the length of the arrays rmatind and rmatval.
    int rmatbeg[rcnt];
    int rmatind[nzcnt];
    double rmatval[nzcnt];
    double rhs[rcnt];
    char sense[rcnt];
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
    int status = solverAddRows(env, model, rcnt, nzcnt, rhs, sense, rmatbeg, rmatind, rmatval, rowname);

    for (int r = 0; r < rcnt; r++)
		delete[] rowname[r];
    delete[] rowname;

    return status;
}

int SolverBilevelShuffleSplit::addOverageConstrs(void *env, void *model)
{
    // Add constraints related to overage costs
    int rcnt = myData->splitSize * myData->nbFolds; // An integer that indicates the number of new rows to be added to the constraint matrix.
    int ncolsperrow = myData->nbFeatures + 1;
    int nzcnt = rcnt * ncolsperrow; // An integer that indicates the number of nonzero constraint coefficients to be added to the constraint matrix. This specifies the length of the arrays rmatind and rmatval.
    int rmatbeg[rcnt];
    int rmatind[nzcnt];
    double rmatval[nzcnt];
    double rhs[rcnt];
    char sense[rcnt];
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
    int status = solverAddRows(env, model, rcnt, nzcnt, rhs, sense, rmatbeg, rmatind, rmatval, rowname);

    for (int r = 0; r < rcnt; r++)
		delete[] rowname[r];
    delete[] rowname;

    return status;
}

int SolverBilevelShuffleSplit::addBetaIndConstrs(void *env, void *model, const int startBeta, const int startZ)
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
                status = solverAddIndConstr(env, model, indvar, true, 1, 0, 'E', &ind, &val, indname_str);
                delete[] indname_str;
                if (status) return status;
            }
        }
    }
    return status;
}

int SolverBilevelShuffleSplit::addFollowerOptConstr(void *env, void *model)
{
    // Add constraint related to the lower-level problem's optimality condition    
    int rcnt = myData->nbFolds; // An integer that indicates the number of new rows to be added to the constraint matrix.
    int nzcnt = rcnt * (myData->splitTrainSize * 4);
    int rmatbeg[rcnt];
    int rmatind[nzcnt];
    double rmatval[nzcnt];
    double rhs[rcnt];
    char sense[rcnt];
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
    int status = solverAddRows(env, model, rcnt, nzcnt, rhs, sense, rmatbeg, rmatind, rmatval, rowname);

    for (int r = 0; r < rcnt; r++)
		delete[] rowname[r];
    delete[] rowname;

    return status;
}


int SolverBilevelShuffleSplit::addDualMuConstrs(void *env, void *model, const int startMu)
{
    // Add constraints related to dual variable mu
    int rcnt, nzcnt;
    rcnt = nzcnt = myData->nbFolds * myData->splitTrainSize; 
    int rmatbeg[rcnt];
    int rmatind[nzcnt];
    double rmatval[nzcnt];
    double rhs[rcnt];
    char sense[rcnt];
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
    int status = solverAddRows(env, model, rcnt, nzcnt, rhs, sense, rmatbeg, rmatind, rmatval, rowname);

    for (int i = 0; i < rcnt; i++)
		delete[] rowname[i];
    delete[] rowname;

    return status;
}

int SolverBilevelShuffleSplit::addDualGammaConstrs(void *env, void *model, const int startGamma)
{
    // Add constraints related to dual variable gamma
    int rcnt, nzcnt;
    rcnt = nzcnt = myData->nbFolds * myData->splitTrainSize;
    int rmatbeg[rcnt];
    int rmatind[nzcnt];
    double rmatval[nzcnt];
    double rhs[rcnt];
    char sense[rcnt];
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
    int status = solverAddRows(env, model, rcnt, nzcnt, rhs, sense, rmatbeg, rmatind, rmatval, rowname);

    for (int i = 0; i < rcnt; i++)
		delete[] rowname[i];
    delete[] rowname;

    return status;
}

int SolverBilevelShuffleSplit::addDualIndConstrs(void *env, void *model)
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
            int linind[nzcnt];
            double linval[nzcnt];
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
                status = solverAddIndConstr(env, model, indvar, false, nzcnt, 0, 'E', linind, linval, indname_str);
                if (status) return status;
            }
            delete[] indname_str;
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
        fprintf(stderr, "Failed to initialize model\n");
        return status;
    }

    // ----------------------------------------------------------------------------
    // Add variables to the model
    int sizeU, sizeO, sizeBeta, sizeMu, sizeGamma, sizeZ;

    // Create columns related to the u variables
    if (status = addUnderageVariables(env, model, sizeU)) return status;

    // Create columns related to the o variables
    if (status = addOverageVariables(env, model, sizeO)) return status;

    // Create columns related to the beta variables
    if (status = addBetaVariables(env, model, sizeBeta)) return status;

    // Create columns related to the mu variables
    if (status = addDualMuVariables(env, model, sizeMu)) return status;

    // Create columns related to the gamma variables
    if (status = addDualGammaVariables(env, model, sizeGamma)) return status;

    // Create columns related to the z variables
    if (status = addZVariables(env, model, sizeZ)) return status;

    int sizeVars = sizeU + sizeO + sizeBeta + sizeMu + sizeGamma + sizeZ;

    if (debug)
        std::cout << "Added all columns - OK" << endl;

    // ----------------------------------------------------------------------------
    // Add constraints to the model
    
    // Add constraints related to underage costs
    if (status = addUnderageConstrs(env, model)) return status;

    // Add constraints related to overage costs
    if (status = addOverageConstrs(env, model)) return status;

    // Add indicator constraints related to the beta and z variables
    if (status = addBetaIndConstrs(env, model, sizeU+sizeO, sizeU+sizeO+sizeBeta+sizeMu+sizeGamma)) return status;

    // Add constraint related to the lower-level problem's optimality condition
    if (status = addFollowerOptConstr(env, model)) return status;

    // Add constraints related to dual variable mu
    if (status = addDualMuConstrs(env, model, sizeU+sizeO+sizeBeta)) return status;

    // Add constraints related to dual variable gamma
    if (status = addDualGammaConstrs(env, model, sizeU+sizeO+sizeBeta+sizeMu)) return status;

    // Add indicator constraints related to the dual variables mu and gamma
    if (status = addDualIndConstrs(env, model)) return status;

    int sizeConstrs = 2*myData->splitSize*myData->nbFolds + myData->nbFolds + 2*myData->splitTrainSize*myData->nbFolds + 2*myData->nbFeatures*myData->nbFolds;

    if (debug)
        std::cout << "Added all constraints - OK" << endl;

    // ----------------------------------------------------------------------------
    // Write problem to file
    if (myData->save_model)
    {
        string ext = ".lp";
        string model_path = mySolution->getOutputFilename(ext);
        status = solverWriteModel(env, model, model_path.c_str()); // Exporting the model
        if (status) return status;
    }

    // Solve MIP
    status = solverOptimize(env, model, MIP_PROBLEM); // Solving the model
    if (status) return status;
    
    // Get the size of the model
    int numcols, numlinconstrs, numindconstrs, numrows;
    status = solverGetModelDimensions(env, model, numcols, numlinconstrs, numindconstrs, numrows);
    if (status) return status;

    printf("Number of columns: %d\n", numcols);
    printf("Number of rows: %d\n", numrows);
    printf("Number of linear constraints: %d\n", numlinconstrs);
    printf("Number of indicator constraints: %d\n", numindconstrs);

    if ((numcols != sizeVars) || (numrows != sizeConstrs))
    {
        fprintf(stderr, "ERROR: There is something wrong with the model dimensions. \n");
        quit_solver(env, model);
        return 1;
    }

    // Retrieve solution
    int lpstat;
    double objval, bestobjval, mipgap, nodecount;
    double solution[sizeVars];
    status = solverRetrieveSolution(env, model, &lpstat, &objval, &bestobjval, &mipgap, &nodecount, solution, sizeVars);
    if (status) return status;

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
    printf("Number of selected features: %d\n", numSelFeats);

    // Create new solution structure
    this->mySolution->update(numcols, numrows, lpstat, stat_str, objval, bestobjval, nodecount, mipgap, solution);

    status = quit_solver(env, model);

    return status;
}

