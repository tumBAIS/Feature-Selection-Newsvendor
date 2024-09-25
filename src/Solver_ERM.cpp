#include "Solver.h"


// ====================================================
// ERM model (without regularization)
// ====================================================

int SolverERM::addUnderageVariables(void *env, void *model, int& size)
{
    // Create columns related to the u variables
    int sizeU = myData->nbSamples;
	double costU[sizeU];
	char **namesU = new char *[sizeU];
	for (int i = 0; i < sizeU; i++)
	{
        costU[i] = (myData->backOrderCost / (double) myData->nbSamples);
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

int SolverERM::addOverageVariables(void *env, void *model, int &size)
{
    // Create columns related to the o variables
    int sizeO = myData->nbSamples;
    double costO[sizeO];
	char **namesO = new char *[sizeO];
	for (int i = 0; i < sizeO; i++)
	{
        costO[i] = (myData->holdingCost / myData->nbSamples);
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

int Solver::addBetaVariables(void *env, void *model, int& size)
{
    // Create columns related to the beta variables
    int sizeBeta = myData->nbFeatures;
	double lbBeta[sizeBeta];
    char **namesBeta = new char *[sizeBeta];
	for (int j = 0; j < myData->nbFeatures; j++)
	{
        lbBeta[j] = -CPX_INFBOUND;
        namesBeta[j] = new char[100];
		sprintf(namesBeta[j], "beta_%d", j);
	}
    int status = solverAddCols(env, model, sizeBeta, NULL, lbBeta, NULL, NULL, namesBeta);

    for (int j = 0; j < sizeBeta; j++)
		delete[] namesBeta[j];
    delete[] namesBeta;

    size = sizeBeta;
    return status;
}

int Solver::addUnderageConstrs(void *env, void *model)
{
    // Add constraints related to underage costs
    int rcnt = myData->nbSamples; // An integer that indicates the number of new rows to be added to the constraint matrix.
    int ncolsperrow = (myData->nbFeatures + 1);
    int nzcnt = rcnt * ncolsperrow; // An integer that indicates the number of nonzero constraint coefficients to be added to the constraint matrix. This specifies the length of the arrays rmatind and rmatval.
    int rmatbeg[rcnt];
    int rmatind[nzcnt];
    double rmatval[nzcnt];
    double rhs[rcnt];
    char sense[rcnt];
    char **rowname = new char *[rcnt];
    for (int i=0; i<myData->nbSamples; i++) 
    {
        rmatbeg[i] = i*ncolsperrow;
        rmatind[i*ncolsperrow] = i;
        rmatval[i*ncolsperrow] = 1; 
        for (int j=0; j<myData->nbFeatures; j++)
        {
            rmatind[i*ncolsperrow+j+1] = 2*myData->nbSamples + j;
            rmatval[i*ncolsperrow+j+1] = myData->feature_data[i][j];
        }
        rhs[i] = myData->demand[i];
        sense[i] = 'G';
        rowname[i] = new char[100];
        sprintf(rowname[i], "underage(%d)", i);
    }
    int error = solverAddRows(env, model, rcnt, nzcnt, rhs, sense, rmatbeg, rmatind, rmatval, rowname);

    for (int i = 0; i < rcnt; i++)
		delete[] rowname[i];
    delete[] rowname;

    return error;
}

int Solver::addOverageConstrs(void *env, void *model)
{
    // Add constraints related to overage costs
    int rcnt = myData->nbSamples; 
    int ncolsperrow = (myData->nbFeatures + 1);
    int nzcnt = rcnt * ncolsperrow; 
    int rmatbeg[rcnt];
    int rmatind[nzcnt];
    double rmatval[nzcnt];
    double rhs[rcnt];
    char sense[rcnt];
    char **rowname = new char *[rcnt];
    for (int i=0; i<myData->nbSamples; i++) 
    {
        rmatbeg[i] = i*ncolsperrow;
        rmatind[i*ncolsperrow] = myData->nbSamples + i;
        rmatval[i*ncolsperrow] = 1; 
        for (int j=0; j<myData->nbFeatures; j++)
        {
            rmatind[i*ncolsperrow+j+1] = 2*myData->nbSamples + j;
            rmatval[i*ncolsperrow+j+1] = - myData->feature_data[i][j];
        }
        rhs[i] = - myData->demand[i];
        sense[i] = 'G';
        rowname[i] = new char[100];
        sprintf(rowname[i], "overage(%d)", i);
    }
    int error = solverAddRows(env, model, rcnt, nzcnt, rhs, sense, rmatbeg, rmatind, rmatval, rowname);

    for (int i = 0; i < rcnt; i++)
		delete[] rowname[i];
    delete[] rowname;

    return error;
}

int Solver::addBetaIndConstrs(void *env, void *model, const int startBeta, const int startZ)
{
    // Add indicator constraints related to the beta and z variables
    int error = 0;
    int indvar;
    int ind;
    double val;
    char *indname_str;
    for (int j=0; j<myData->nbFeatures; j++)
	{
        indvar = startZ + j;
		ind = startBeta + j;
        val = 1;
        indname_str = new char[100];
        sprintf(indname_str, "indBeta(%d)", j);
        error = solverAddIndConstr(env, model, indvar, true, 1, 0, 'E', &ind, &val, indname_str);
        delete[] indname_str;
        if (error) return error;
	}

    return error;
}

// ================================================================= //


int SolverERM::solve()
{
    int error = 0;

    std::cout << "Solving optimization problem" << endl;

    // Initialize model
    error = initializeModel(1, "Newsvendor-ERM");
    if (error) {
        fprintf(stderr, "Failed to initialize model\n");
        return error;
    }

    // Create columns related to the u variables
    int sizeU;
    if (error = addUnderageVariables(env, model, sizeU)) return error;

    // Create columns related to the o variables
    int sizeO;
    if (error = addOverageVariables(env, model, sizeO)) return error;

    // Create columns related to the beta variables
    int sizeBeta;
    if (error = addBetaVariables(env, model, sizeBeta)) return error;

    // Total number of variables
    int sizeVars = sizeU + sizeO + sizeBeta;

    // Add constraints related to underage costs
    if (error = addUnderageConstrs(env, model)) return error;

    // Add constraints related to overage costs
    if (error = addOverageConstrs(env, model)) return error;

    // Total number of constraints
    int sizeConstrs = 2*myData->nbSamples;

    // Write problem to file
    if (myData->save_model)
    {
        string ext = ".lp";
        string model_path = mySolution->getOutputFilename(ext);
        error = solverWriteModel(env, model, model_path.c_str()); // Exporting the model
        if (error) return error;
    }

    // Solve LP
    error = solverOptimize(env, model, LP_PROBLEM); // Solving the model
    if (error) return error;

    // Get the size of the model
    int numcols, numrows, cur_numlconstrs, cur_numgenconstrs;
    error = solverGetModelDimensions(env, model, numcols, cur_numlconstrs, cur_numgenconstrs, numrows);
    if (error) return error;

    std::cout << "Number of columns: " << numcols << std::endl;
    std::cout << "Number of rows: " << numrows << std::endl;
    std::cout << "Number of linear constraints: " << cur_numlconstrs << std::endl;
    std::cout << "Number of indicator constraints: " << cur_numgenconstrs << std::endl;

    if ((numcols != sizeVars) || (numrows != sizeConstrs))
    {
        fprintf(stderr, "ERROR: There is something wrong with the model. \n");
        quit_solver(env, model);
        return 1;
    }

    // Retrieve solution
    int lpstat;
    double objval;
    double solution[sizeVars];
    error = solverRetrieveSolution(env, model, &lpstat, &objval, NULL, NULL, NULL, solution, sizeVars);
    if (error) return error;

    // Convert status to string 
    string stat_str = getSolverStatusString(lpstat);

    // Print results
    std::cout << std::endl;
    std::cout << "Solution status:                   " << lpstat << " (" << stat_str << ")" << std::endl;
    std::cout << "Objective value:                   " << std::setprecision(10) << objval << std::endl;
    std::cout << endl;

    std::cout << std::setw(10) << "beta" << endl; 
    for (int j=0; j<myData->nbFeatures; j++)
    {
        std::cout << std::setw(10) << solution[sizeU+sizeO+j] << endl;
    }

    // Create new solution structure
    mySolution->update(numcols, numrows, lpstat, stat_str, objval, 0, 0, 0.0, solution);

    error = quit_solver(env, model);

    return error;
}



// ====================================================
// ERM with l0-norm regularization
// ====================================================

int SolverERM_l0::solve()
{
    int error = 0;

    double lambda = myData->regularizationParam;

    std::cout << "Solving optimization problem" << endl;

    std::cout << "Lambda: " << lambda << endl;

    // Initialize model
    error = initializeModel(1, "Newsvendor-ERM");
    if (error) {
        fprintf(stderr, "Failed to initialize model\n");
        return error;
    }

    // Create columns related to the u variables
    int sizeU;
    if (error = addUnderageVariables(env, model, sizeU)) return error;

    // Create columns related to the o variables
    int sizeO;
    if (error = addOverageVariables(env, model, sizeO)) return error;

    // Create columns related to the beta variables
    int sizeBeta;
    if (error = addBetaVariables(env, model, sizeBeta)) return error;

    // Create columns related to the z variables
    int sizeZ = myData->nbFeatures;
    double costZ[sizeZ];
    char xctypeZ[sizeZ];
    char **namesZ = new char *[sizeZ];
	for (int j = 0; j < myData->nbFeatures; j++)
	{
        costZ[j] = lambda; 
        xctypeZ[j] = 'B';
        namesZ[j] = new char[100];
		sprintf(namesZ[j], "z_%d", j);
	}
    error = solverAddCols(env, model, sizeZ, costZ, NULL, NULL, xctypeZ, namesZ);
    if (error) return error;

    for (int j = 0; j < sizeZ; j++)
		delete[] namesZ[j];
    delete[] namesZ;

    int sizeVars = sizeU + sizeO + sizeBeta + sizeZ;

    // Add constraints related to underage costs
    if (error = addUnderageConstrs(env, model)) return error;

    // Add constraints related to overage costs
    if (error = addOverageConstrs(env, model)) return error;

    // Add indicator constraints related to the beta variables
    if (error = addBetaIndConstrs(env, model, sizeU+sizeO, sizeU+sizeO+sizeBeta)) return error;

    int sizeConstrs = 2*myData->nbSamples + myData->nbFeatures;
    
    // Write problem to file
    if (myData->save_model)
    {
        string ext = ".lp";
        string model_path = mySolution->getOutputFilename(ext);
        error = solverWriteModel(env, model, model_path.c_str()); // Exporting the model
        if (error) return error;
    }

    // Solve MILP
    error = solverOptimize(env, model, MIP_PROBLEM); // Solving the model
    if (error) return error;

    // Get the size of the model
    int numcols, numrows, cur_numlconstrs, cur_numgenconstrs;
    error = solverGetModelDimensions(env, model, numcols, cur_numlconstrs, cur_numgenconstrs, numrows);
    if (error) return error;

    std::cout << "Number of columns: " << numcols << std::endl;
    std::cout << "Number of rows: " << numrows << std::endl;
    std::cout << "Number of linear constraints: " << cur_numlconstrs << std::endl;
    std::cout << "Number of indicator constraints: " << cur_numgenconstrs << std::endl;

    if ((numcols != sizeVars) || (numrows != sizeConstrs))
    {
        fprintf(stderr, "ERROR: There is something wrong with the model. \n");
        quit_solver(env, model);
        return 1;
    }

    // Retrieve solution
    int lpstat;
    double objval, bestobjval, mipgap, nodecount;
    double solution[sizeVars];
    error = solverRetrieveSolution(env, model, &lpstat, &objval, &bestobjval, &mipgap, &nodecount, solution, sizeVars);
    if (error) return error;

    // Convert status to string 
    string stat_str = getSolverStatusString(lpstat);

    // Print results
    std::cout << std::endl;
    std::cout << "Solution status:                   " << lpstat << " (" << stat_str << ")" << std::endl;
    std::cout << "Objective value:                   " << std::setprecision(10) << objval << std::endl;
    std::cout << "Objective lower bound:             " << std::setprecision(10) << bestobjval << std::endl;
    std::cout << "MIP Relative objective gap:        " << std::setprecision(10) << mipgap << std::endl;
    std::cout << "Nodes processed:                   " << static_cast<int>(nodecount) << std::endl;
    std::cout << endl;

    std::cout << std::setw(10) << "beta" << std::setw(5) << "z" << endl; 
    for (int j=0; j<myData->nbFeatures; j++)
    {
        std::cout << std::setw(10) << solution[sizeU+sizeO+j] << std::setw(5) << solution[sizeU+sizeO+sizeBeta+j] << endl;
    }
    
    // Create new solution structure
    this->mySolution->update(numcols, numrows, lpstat, stat_str, objval, bestobjval, nodecount, mipgap, solution);

    error = quit_solver(env, model);
    
    return error;
}




// ====================================================
// ERM with l1-norm regularization
// ====================================================


int SolverERM_l1::solve()
{
    int error = 0;

    double lambda = myData->regularizationParam;

    std::cout << "Solving optimization problem" << endl;

    std::cout << "Lambda: " << lambda << endl;

    // Initialize model
    error = initializeModel(1, "Newsvendor-ERM");
    if (error) {
        fprintf(stderr, "Failed to initialize model\n");
        return error;
    }

    // Create columns related to the u variables
    int sizeU;
    if (error = addUnderageVariables(env, model, sizeU)) return error;

    // Create columns related to the o variables
    int sizeO;
    if (error = addOverageVariables(env, model, sizeO)) return error;

    // Create columns related to the beta+ variables
    const int sizeBeta_pos = myData->nbFeatures;
    {
        double costBeta_pos[myData->nbFeatures];
        char **namesBeta_pos = new char *[myData->nbFeatures];
        for (int j = 0; j < myData->nbFeatures; j++)
        {
            costBeta_pos[j] = lambda;
            namesBeta_pos[j] = new char[100];
            sprintf(namesBeta_pos[j], "beta_pos_%d", j);
        }
        error = solverAddCols(env, model, sizeBeta_pos, costBeta_pos, NULL, NULL, NULL, namesBeta_pos);
        if (error) return error;

        for (int j = 0; j < sizeBeta_pos; j++)
            delete[] namesBeta_pos[j];
        delete[] namesBeta_pos;
    }

    // Create columns related to the beta- variables
    const int sizeBeta_neg = myData->nbFeatures;
    {
        double costBeta_neg[myData->nbFeatures];
        char **namesBeta_neg = new char *[myData->nbFeatures];
        for (int j = 0; j < myData->nbFeatures; j++)
        {
            costBeta_neg[j] = lambda;
            namesBeta_neg[j] = new char[100];
            sprintf(namesBeta_neg[j], "beta_neg_%d", j);
        }
        error = solverAddCols(env, model, sizeBeta_neg, costBeta_neg, NULL, NULL, NULL, namesBeta_neg);
        if (error) return error;

        for (int j = 0; j < sizeBeta_neg; j++)
            delete[] namesBeta_neg[j];
        delete[] namesBeta_neg;
    }

    int sizeVars = sizeU + sizeO + sizeBeta_pos + sizeBeta_neg;

    // Add constraints related to underage costs
    {
        const int rcnt = myData->nbSamples; // An integer that indicates the number of new rows to be added to the constraint matrix.
        const int ncolsperrow = (myData->nbFeatures*2 + 1);
        const int nzcnt = rcnt * ncolsperrow; // An integer that indicates the number of nonzero constraint coefficients to be added to the constraint matrix. This specifies the length of the arrays rmatind and rmatval.
        int rmatbeg[rcnt];
        int rmatind[nzcnt];
        double rmatval[nzcnt];
        double rhs[rcnt];
        char sense[rcnt];
        char **rowname = new char *[rcnt];
        for (int i=0; i<myData->nbSamples; i++) 
        {
            rmatbeg[i] = i*ncolsperrow;
            rmatind[i*ncolsperrow] = i;
            rmatval[i*ncolsperrow] = 1; 
            for (int j=0; j<myData->nbFeatures; j++) 
            {
                // coefficients for beta_+
                rmatind[i*ncolsperrow+j+1] = sizeU + sizeO + j;
                rmatval[i*ncolsperrow+j+1] = myData->feature_data[i][j];
                // coefficients for beta_-
                rmatind[i*ncolsperrow+myData->nbFeatures+j+1] = sizeU + sizeO + sizeBeta_pos + j;
                rmatval[i*ncolsperrow+myData->nbFeatures+j+1] = -myData->feature_data[i][j];
            }
            rhs[i] = myData->demand[i];
            sense[i] = 'G';
            rowname[i] = new char[100];
            sprintf(rowname[i], "underage(%d)", i);
        }
        error = solverAddRows(env, model, rcnt, nzcnt, rhs, sense, rmatbeg, rmatind, rmatval, rowname);
        if (error) return error;

        for (int i = 0; i < rcnt; i++)
            delete[] rowname[i];
        delete[] rowname;
    }

    // Add constraints related to overage costs
    {
        const int rcnt = myData->nbSamples; 
        const int ncolsperrow = (myData->nbFeatures*2 + 1);
        const int nzcnt = rcnt * ncolsperrow; 
        int rmatbeg[rcnt];
        int rmatind[nzcnt];
        double rmatval[nzcnt];
        double rhs[rcnt];
        char sense[rcnt];
        char **rowname = new char *[rcnt];
        for (int i=0; i<myData->nbSamples; i++) 
        {
            rmatbeg[i] = i*ncolsperrow;
            rmatind[i*ncolsperrow] = sizeU + i;
            rmatval[i*ncolsperrow] = 1; 
            for (int j=0; j<myData->nbFeatures; j++)
            {
                // coefficients for beta_+
                rmatind[i*ncolsperrow+j+1] = sizeU + sizeO + j;
                rmatval[i*ncolsperrow+j+1] = - myData->feature_data[i][j];
                // coefficients for beta_-
                rmatind[i*ncolsperrow+myData->nbFeatures+j+1] = sizeU + sizeO + sizeBeta_pos + j;
                rmatval[i*ncolsperrow+myData->nbFeatures+j+1] = + myData->feature_data[i][j];
            }
            rhs[i] = - myData->demand[i];
            sense[i] = 'G';
            rowname[i] = new char[100];
            sprintf(rowname[i], "overage(%d)", i);
        }
        error = solverAddRows(env, model, rcnt, nzcnt, rhs, sense, rmatbeg, rmatind, rmatval, rowname);
        if (error) return error;

        for (int i = 0; i < rcnt; i++)
            delete[] rowname[i];
        delete[] rowname;
    }

    int sizeConstrs = 2*myData->nbSamples;

    // Write problem to file
    if (myData->save_model)
    {
        string ext = ".lp";
        string model_path = mySolution->getOutputFilename(ext);
        error = solverWriteModel(env, model, model_path.c_str()); // Exporting the model
        if (error) return error;
    }

    // Solve LP
    error = solverOptimize(env, model, LP_PROBLEM); // Solving the model
    if (error) return error;

    // Get the size of the model
    int numcols, numrows, cur_numlconstrs, cur_numgenconstrs;
    error = solverGetModelDimensions(env, model, numcols, cur_numlconstrs, cur_numgenconstrs, numrows);
    if (error) return error;

    std::cout << "Number of columns: " << numcols << std::endl;
    std::cout << "Number of rows: " << numrows << std::endl;
    std::cout << "Number of linear constraints: " << cur_numlconstrs << std::endl;
    std::cout << "Number of indicator constraints: " << cur_numgenconstrs << std::endl;
    
    if ((numcols != sizeVars) || (numrows != sizeConstrs))
    {
        fprintf(stderr, "ERROR: There is something wrong with the model. \n");
        quit_solver(env, model);
        return 1;
    }

    // Retrieve solution
    int lpstat;
    double objval;
    double solution[sizeVars];
    error = solverRetrieveSolution(env, model, &lpstat, &objval, NULL, NULL, NULL, solution, sizeVars);
    if (error) return error;

    // Convert status to string 
    string stat_str = getSolverStatusString(lpstat);

    // Print results
    std::cout << std::endl;
    std::cout << "Solution status:                   " << lpstat << " (" << stat_str << ")" << std::endl;
    std::cout << "Objective value:                   " << std::setprecision(10) << objval << std::endl;
    std::cout << endl;

    int w = 10;
    std::cout << std::setw(w) << "beta" << std::setw(w) << "beta_+" << std::setw(w) << "beta_-" << endl; 
    for (int j=0; j<myData->nbFeatures; j++)
    {
        std::cout << std::setw(w) << solution[sizeU+sizeO+j] - solution[sizeU+sizeO+sizeBeta_pos+j];
        std::cout << std::setw(w) << solution[sizeU+sizeO+j];
        std::cout << std::setw(w) << solution[sizeU+sizeO+sizeBeta_pos+j] << endl;
    }

    // Create new solution structure
    mySolution->update(numcols, numrows, lpstat, stat_str, objval, 0, 0, 0.0, solution);

    error = quit_solver(env, model);

    return error;
}


