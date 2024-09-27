#include "Solver.h"

int SolverSubProblem::addUnderageVariables(int& size)
{
    // Create columns related to the u variables
    int sizeU = myData->nbTrainSamples;
	double *costU = new double[sizeU];
	char **namesU = new char *[sizeU];
    int r = 0;
	for (int i = 0; i < myData->nbSamples; i++)
	{
        if (myData->holdoutSplit[i] == 1) {
            costU[r] = (myData->backOrderCost / myData->nbTrainSamples);
            namesU[r] = new char[100];
            sprintf(namesU[r], "u_%d", i);
            r++;
        }
	}
    int status = solverAddCols(sizeU, costU, NULL, NULL, NULL, namesU);
    
    delete[] costU;
    for (int r = 0; r < sizeU; r++)
		delete[] namesU[r];
    delete[] namesU;

    size = sizeU; // return number of variables added to the model

    return status;    
}


int SolverSubProblem::addOverageVariables(int& size)
{
    // Create columns related to the o variables
    int sizeO = myData->nbTrainSamples;
    double *costO = new double[sizeO];
	char **namesO = new char *[sizeO];
    int r = 0;
	for (int i = 0; i < myData->nbSamples; i++)
	{
        if (myData->holdoutSplit[i] == 1) {
            costO[r] = (myData->holdingCost / myData->nbTrainSamples);
            namesO[r] = new char[100];
            sprintf(namesO[r], "o_%d", i);
            r++;
        }
	}
    int status = solverAddCols(sizeO, costO, NULL, NULL, NULL, namesO);
    
    delete[] costO;
    for (int r = 0; r < sizeO; r++)
		delete[] namesO[r];
    delete[] namesO;

    size = sizeO; // return number of variables added to the model

    return status;
}


int SolverSubProblem_ERM_l0::addUnderageConstrs()
{
    // Add constraints related to underage costs
    int rcnt = myData->nbTrainSamples; // An integer that indicates the number of new rows to be added to the constraint matrix.
    int ncolsperrow = (myData->nbFeatures + 1);
    int nzcnt = rcnt * ncolsperrow; // An integer that indicates the number of nonzero constraint coefficients to be added to the constraint matrix. This specifies the length of the arrays rmatind and rmatval.
    int rmatbeg[rcnt];
    int rmatind[nzcnt];
    double rmatval[nzcnt];
    double rhs[rcnt];
    char sense[rcnt];
    char **rowname = new char *[rcnt];
    int r = 0;
    for (int i=0; i<myData->nbSamples; i++) 
    {
        if (myData->holdoutSplit[i] == 1) {
            rmatbeg[r] = r*ncolsperrow;
            rmatind[r*ncolsperrow] = r;
            rmatval[r*ncolsperrow] = 1; 
            for (int j=0; j<myData->nbFeatures; j++)
            {
                rmatind[r*ncolsperrow+j+1] = 2*myData->nbTrainSamples + j;
                rmatval[r*ncolsperrow+j+1] = myData->feature_data[i][j];
            }
            rhs[r] = myData->demand[i];
            sense[r] = 'G';
            rowname[r] = new char[100];
            sprintf(rowname[r], "underage(%d)", i);
            r++;
        }
    }
    int status = solverAddRows(rcnt, nzcnt, rhs, sense, rmatbeg, rmatind, rmatval, rowname);

    for (int r = 0; r < rcnt; r++)
		delete[] rowname[r];
    delete[] rowname;

    return status;
}

int SolverSubProblem_ERM_l0::addOverageConstrs()
{
    // Add constraints related to overage costs
    int rcnt = myData->nbTrainSamples; 
    int ncolsperrow = (myData->nbFeatures + 1);
    int nzcnt = rcnt * ncolsperrow; 
    int rmatbeg[rcnt];
    int rmatind[nzcnt];
    double rmatval[nzcnt];
    double rhs[rcnt];
    char sense[rcnt];
    char **rowname = new char *[rcnt];
    int r = 0;
    for (int i=0; i<myData->nbSamples; i++) 
    {
        if (myData->holdoutSplit[i] == 1) {
            rmatbeg[r] = r*ncolsperrow;
            rmatind[r*ncolsperrow] = myData->nbTrainSamples + r;
            rmatval[r*ncolsperrow] = 1; 
            for (int j=0; j<myData->nbFeatures; j++)
            {
                rmatind[r*ncolsperrow+j+1] = 2*myData->nbTrainSamples + j;
                rmatval[r*ncolsperrow+j+1] = - myData->feature_data[i][j];
            }
            rhs[r] = - myData->demand[i];
            sense[r] = 'G';
            rowname[r] = new char[100];
            sprintf(rowname[r], "overage(%d)", i);
            r++;
        }
    }
    int status = solverAddRows(rcnt, nzcnt, rhs, sense, rmatbeg, rmatind, rmatval, rowname);

    for (int r = 0; r < rcnt; r++)
		delete[] rowname[r];
    delete[] rowname;

    return status;
}


int SolverSubProblem_ERM_l0::solve()
{
    bool debug = false;
    int error = 0;

    double lambda = myData->regularizationParam; // Regularization parameter

    if (debug) {
        std::cout << endl << "### Solving subproblem ";
        if (spIndex >= 0)
            std::cout << spIndex;
        std::cout << endl;
        if (myData->numActiveFeatures >= 0)
            std::cout << "Active features: " << myData->numActiveFeatures << endl;
        std::cout << "Lambda: " << lambda << endl;
        std::cout << "Time limit: " << myData->timeLimit << endl;
    }

    error = initializeModel(0, "Subproblem-ERM_l0");
    if (error) {
        std::cerr << "Failed to initialize model" << std::endl;
        return error;
    }

    // ----------------------------------------------------------------------------
    // Add variables to the model

    int sizeU, sizeO, sizeBeta;

    // Create columns related to the u variables
    if (error = addUnderageVariables(sizeU)) return error;
    
    // Create columns related to the o variables
    if (error = addOverageVariables(sizeO)) return error;

    // Create columns related to the beta variables
    if (error = addBetaVariables(sizeBeta)) return error;

    // Create columns related to the z variables
    int sizeZ = myData->nbFeatures;
    double *costZ = new double[myData->nbFeatures];
    char vtypeZ[sizeZ];
    char **namesZ = new char *[sizeZ];
	for (int j = 0; j < myData->nbFeatures; j++)
	{
        if (j == 0) {
            costZ[j] = 0.0;     
        } else {
            costZ[j] = lambda; 
        }
        vtypeZ[j] = 'B';
        namesZ[j] = new char[100];
		sprintf(namesZ[j], "z_%d", j);
	}
    error = solverAddCols(sizeZ, costZ, NULL, NULL, vtypeZ, namesZ);
    
    delete[] costZ;
    for (int j = 0; j < sizeZ; j++)
		delete[] namesZ[j];
    delete[] namesZ;

    if (error) return error;

    int sizeVars = sizeU + sizeO + sizeBeta + sizeZ;

    // ----------------------------------------------------------------------------
    // Add constraints to the model

    // Add constraints related to underage costs
    if (error = addUnderageConstrs()) return error;

    // Add constraints related to overage costs
    if (error = addOverageConstrs()) return error;
    
    // Add indicator constraints related to the beta variables
    if (error = addBetaIndConstrs(sizeU+sizeO, sizeU+sizeO+sizeBeta)) return error;

    int sizeConstrs = 2*myData->nbTrainSamples + myData->nbFeatures;

    // Add constraints that enforce the number of active features
    if (myData->numActiveFeatures >= 0)
    {
        int rmatind[myData->nbFeatures];
        double rmatval[myData->nbFeatures];
        char *rowname = new char [100];
        
        int beg = 0;
        double rhs = myData->numActiveFeatures;
        char sense = 'E';
        sprintf(rowname, "num_active_features");

        for (int j=0; j<myData->nbFeatures; j++) 
        {
            rmatind[j] = sizeU + sizeO + sizeBeta + j;
            rmatval[j] = 1; 
        }
        error = solverAddRows(1, myData->nbFeatures, &rhs, &sense, &beg, rmatind, rmatval, &rowname);
        
        delete[] rowname;

        if (error) return error;

        sizeConstrs++;
    }

    // Write problem to file
    if (myData->save_model)
    {
        string ext;
        if (this->spIndex >= 0)
            ext = "-sp" + to_string(this->spIndex) + ".lp";
        else
            ext = ".lp";
        string model_path = mySolution->getOutputFilename(ext);
        error = solverWriteModel(model_path.c_str()); // Exporting the model
        if (error) return error;
    }

    // Solve MIP
    error = solverOptimize(MIP_PROBLEM); // Solving the model
    if (error) return error;

    // Get the size of the model
    int numcols, numrows, cur_numlconstrs, cur_numgenconstrs;
    error = solverGetModelDimensions(numcols, cur_numlconstrs, cur_numgenconstrs, numrows);
    if (error) return error;
    
    if ((numcols != sizeVars) || (numrows != sizeConstrs))
    {
        std::cerr << "ERROR: There is something wrong with the model." << std::endl;
        quit_solver();
        return 1;
    }

    // Retrieve solution
    int lpstat;
    double objval, bestobjval, mipgap, nodecount;
    double solution[sizeVars];
    error = solverRetrieveSolution(&lpstat, &objval, &bestobjval, &mipgap, &nodecount, solution, sizeVars);
    if (error) return error;

    // Convert status to string 
    string stat_str = getSolverStatusString(lpstat);

    // Print results
    if (debug) {
        std::cout << std::endl;
        std::cout << "Solution status:                   " << lpstat << " (" << stat_str << ")" << std::endl;
        std::cout << "Objective value:                   " << std::setprecision(10) << objval << std::endl;
        std::cout << "Objective lower bound:             " << std::setprecision(10) << bestobjval << std::endl;
        std::cout << "MIP Relative objective gap:        " << std::setprecision(10) << mipgap << std::endl;
        std::cout << "Nodes processed:                   " <<  static_cast<int>(nodecount) << std::endl;
    }

    // Create new solution structure
    mySolution->update(numcols, numrows, lpstat, stat_str, objval, bestobjval, nodecount, mipgap, solution);

    error = quit_solver();

    return error;
}




int SolverSubProblem_ERM_l1::addUnderageConstrs()
{
    // Add constraints related to underage costs
    int rcnt = myData->nbTrainSamples; // An integer that indicates the number of new rows to be added to the constraint matrix.
    int ncolsperrow = (myData->nbFeatures*2 + 1);
    int nzcnt = rcnt * ncolsperrow; // An integer that indicates the number of nonzero constraint coefficients to be added to the constraint matrix. This specifies the length of the arrays rmatind and rmatval.
    int rmatbeg[rcnt];
    int rmatind[nzcnt];
    double rmatval[nzcnt];
    double rhs[rcnt];
    char sense[rcnt];
    char **rowname = new char *[rcnt];
    int r = 0;
    for (int i=0; i<myData->nbSamples; i++) 
    {
        if (myData->holdoutSplit[i] == 1) {
            rmatbeg[r] = r*ncolsperrow;
            rmatind[r*ncolsperrow] = r;
            rmatval[r*ncolsperrow] = 1; 
            for (int j=0; j<myData->nbFeatures; j++) 
            {
                // coefficients for beta+
                rmatind[r*ncolsperrow+j+1] = 2*myData->nbTrainSamples + j;
                rmatval[r*ncolsperrow+j+1] = myData->feature_data[i][j];
                // coefficients for beta-
                rmatind[r*ncolsperrow+myData->nbFeatures+j+1] = 2*myData->nbTrainSamples + myData->nbFeatures + j;
                rmatval[r*ncolsperrow+myData->nbFeatures+j+1] = - myData->feature_data[i][j];
            }
            rhs[r] = myData->demand[i];
            sense[r] = 'G';
            rowname[r] = new char[100];
            sprintf(rowname[r], "underage(%d)", i);
            r++;
        }
    }
    int status = solverAddRows(rcnt, nzcnt, rhs, sense, rmatbeg, rmatind, rmatval, rowname);

    for (int r = 0; r < rcnt; r++)
		delete[] rowname[r];
    delete[] rowname;

    return status;
}

int SolverSubProblem_ERM_l1::addOverageConstrs()
{
    // Add constraints related to overage costs
    int rcnt = myData->nbTrainSamples; 
    int ncolsperrow = (myData->nbFeatures*2 + 1);
    int nzcnt = rcnt * ncolsperrow; 
    int rmatbeg[rcnt];
    int rmatind[nzcnt];
    double rmatval[nzcnt];
    double rhs[rcnt];
    char sense[rcnt];
    char **rowname = new char *[rcnt];
    int r = 0;
    for (int i=0; i<myData->nbSamples; i++) 
    {
        if (myData->holdoutSplit[i] == 1) {
            rmatbeg[r] = r*ncolsperrow;
            rmatind[r*ncolsperrow] = myData->nbTrainSamples + r;
            rmatval[r*ncolsperrow] = 1; 
            for (int j=0; j<myData->nbFeatures; j++)
            {
                // coefficients for beta_+
                rmatind[r*ncolsperrow+j+1] = 2*myData->nbTrainSamples + j;
                rmatval[r*ncolsperrow+j+1] = - myData->feature_data[i][j];
                // coefficients for beta_-
                rmatind[r*ncolsperrow+myData->nbFeatures+j+1] = 2*myData->nbTrainSamples + myData->nbFeatures + j;
                rmatval[r*ncolsperrow+myData->nbFeatures+j+1] = + myData->feature_data[i][j];
            }
            rhs[r] = - myData->demand[i];
            sense[r] = 'G';
            rowname[r] = new char[100];
            sprintf(rowname[r], "overage(%d)", i);
            r++;
        }
    }
    int status = solverAddRows(rcnt, nzcnt, rhs, sense, rmatbeg, rmatind, rmatval, rowname);

    for (int i = 0; i < rcnt; i++)
		delete[] rowname[i];
    delete[] rowname;
    
    return status;
}

int SolverSubProblem_ERM_l1::solve()
{
    bool debug = false;
    int error = 0;

    double lambda = myData->regularizationParam; // Regularization parameter

    if (debug) {
        std::cout << endl << "### Solving subproblem ";
        if (spIndex >= 0)
            std::cout << spIndex;
        std::cout << endl;
        std::cout << "Lambda: " << lambda << endl;
        std::cout << "Time limit: " << myData->timeLimit << endl;
    }

    error = initializeModel(0, "Subproblem-ERM_l1");
    if (error) {
        std::cerr << "Failed to initialize model" << std::endl;
        return error;
    }

    // ----------------------------------------------------------------------------
    // Add variables to the model

    int sizeU, sizeO;

    // Create columns related to the u variables
    if (error = addUnderageVariables(sizeU)) return error;
    
    // Create columns related to the o variables
    if (error = addOverageVariables(sizeO)) return error;

    // Create columns related to the beta+ variables
    {
        int sizeBeta_pos = myData->nbFeatures;
        double costBeta_pos[sizeBeta_pos];
        char **namesBeta_pos = new char *[sizeBeta_pos];
        for (int j = 0; j < myData->nbFeatures; j++)
        {
            if (j == 0) {
                costBeta_pos[j] = 0.0;
            } else {
                costBeta_pos[j] = lambda;
            }
            namesBeta_pos[j] = new char[100];
            sprintf(namesBeta_pos[j], "beta_pos_%d", j);
        }
        error = solverAddCols(sizeBeta_pos, costBeta_pos, NULL, NULL, NULL, namesBeta_pos);
        
        for (int j = 0; j < sizeBeta_pos; j++)
            delete[] namesBeta_pos[j];
        delete[] namesBeta_pos;

        if (error) return error;
    }

    // Create columns related to the beta- variables
    {
        int sizeBeta_neg = myData->nbFeatures;
        double costBeta_neg[sizeBeta_neg];
        char **namesBeta_neg = new char *[sizeBeta_neg];
        for (int j = 0; j < myData->nbFeatures; j++)
        {
            if (j == 0) {
                costBeta_neg[j] = 0.0;
            } else {
                costBeta_neg[j] = lambda;
            }
            namesBeta_neg[j] = new char[100];
            sprintf(namesBeta_neg[j], "beta_neg_%d", j);
        }
        error = solverAddCols(sizeBeta_neg, costBeta_neg, NULL, NULL, NULL, namesBeta_neg);
        
        for (int j = 0; j < sizeBeta_neg; j++)
            delete[] namesBeta_neg[j];
        delete[] namesBeta_neg;

        if (error) return error;
    }

    int sizeVars = 2*myData->nbTrainSamples + 2*myData->nbFeatures;

    // ----------------------------------------------------------------------------
    // Add constraints to the model

    // Add constraints related to underage costs
    if (error = addUnderageConstrs()) return error;

    // Add constraints related to overage costs
    if (error = addOverageConstrs()) return error;

    int sizeConstrs = 2*myData->nbTrainSamples;

    // Write problem to file
    if (myData->save_model)
    {
        string ext;
        if (this->spIndex >= 0)
            ext = "-sp" + to_string(this->spIndex) + ".lp";
        else
            ext = ".lp";
        string model_path = mySolution->getOutputFilename(ext);
        error = solverWriteModel(model_path.c_str()); // Exporting the model
        if (error) return error;
    }

    // Solve MIP
    error = solverOptimize(LP_PROBLEM); // Solving the model
    if (error) return error;

    // Get the size of the model
    int numcols, numrows, cur_numlconstrs, cur_numgenconstrs;
    error = solverGetModelDimensions(numcols, cur_numlconstrs, cur_numgenconstrs, numrows);
    if (error) return error;
    
    if ((numcols != sizeVars) || (numrows != sizeConstrs))
    {
        std::cerr << "ERROR: There is something wrong with the model." << std::endl;
        quit_solver();
        return 1;
    }

    // Retrieve solution
    int lpstat;
    double objval, nodecount;
    double bestobjval = -1;
    double solution[sizeVars];
    error = solverRetrieveSolution(&lpstat, &objval, NULL, NULL, &nodecount, solution, sizeVars);
    if (error) return error;

    if (isSolOpt(lpstat))
        bestobjval = objval;

    // Convert status to string 
    string stat_str = getSolverStatusString(lpstat);

    // Print results
    if (debug) {
        std::cout << std::endl;
        std::cout << "Solution status:                   " << lpstat << " (" << stat_str << ")" << std::endl;
        std::cout << "Objective value:                   " << std::setprecision(10) << objval << std::endl;
        std::cout << "Nodes processed:                   " << static_cast<int>(nodecount) << std::endl;
    }

    // Create new solution structure
    this->mySolution->update(numcols, numrows, lpstat, stat_str, objval, bestobjval, nodecount, 0, solution);

    error = quit_solver();

    return error;

}


int SolverGridSearch::solve()
{
    int status = 0;
    bool debug = false;

    std::cout << "Solving optimization problem" << endl;

    clock_t time_start = clock(); // Begin of clock
    double time_elapsed = 0;

    double origTimeLimit = myData->timeLimit; // save original time limit

    vector<double> lambdaVector = logspace(-3, 2, myData->nbBreakpoints, 10); // generate breakpoints spaced evenly on a log scale
    lambdaVector.insert(lambdaVector.begin(), 0.0); myData->nbBreakpoints++; // include zero as a breakpoint

    std::shared_ptr<SolutionGridSearch> mySolutionGS = std::dynamic_pointer_cast<SolutionGridSearch>(mySolution);
    vector<std::shared_ptr<SolutionSubProblem>>& spSolutions = mySolutionGS->spSolutions;

    // Initialize models for each breakpoint
    for (int l=0; l<myData->nbBreakpoints; l++)
    {
        spSolutions.push_back(SolutionSubProblem::createSolutionSP(myData));
        spSolvers.push_back(SolverSubProblem::createSolverSP(myData, spSolutions[l]));
        spSolvers[l]->spIndex = l;
        spSolutions[l]->trainCost = CPX_INFBOUND; // Initialize training cost to infinity
        spSolutions[l]->valCost   = CPX_INFBOUND; // Initialize validation cost to infinity
        spSolutions[l]->strStatus = "NOT_SOLVED"; // Initialize status as "NOT_SOLVED"
        mySolutionGS->lambdaVector.push_back(lambdaVector[l]);
    }
    // Initialize final model
    mySolutionGS->finalSolution = SolutionSubProblem::createSolutionSP(myData);
    std::shared_ptr<SolutionSubProblem> finalSolution = mySolutionGS->finalSolution;
    finalSolver   = SolverSubProblem::createSolverSP(myData, finalSolution);

    int opt_lambda_index = -1; // Optimal number of active features
    double opt_val_cost = CPX_INFBOUND; // Validation cost of optimal solution

    // Record if subproblem solutions are optimal
    bool allOpt = true;

    // Perform grid search
    for (int l=0; l<myData->nbBreakpoints; l++)
    {
        time_elapsed = (((double)(clock() - time_start)) / CLOCKS_PER_SEC);
        if (debug)
            std::cout << "Elapsed time: " << time_elapsed << endl;
        if (time_elapsed > origTimeLimit) {
            allOpt = false;
            break;
        }
        myData->timeLimit = std::max(5e-2, origTimeLimit - time_elapsed); // if time limit was reached, set it to very small value

        myData->regularizationParam = lambdaVector[l];  // Set regularization parameter 

        status = spSolvers[l]->solve(); // Solve subproblem l
        if (status) {
            std::cerr << "Failed to solve subproblem " << l << std::endl;
        }

        // Update optimal validation cost and the corresponding number of features
        bool isOptimal = isSolOpt(spSolutions[l]->solverStatus);
        if ((isOptimal == true) && (spSolutions[l]->valCost < opt_val_cost - 1e-5)) {
            opt_lambda_index = l;
            opt_val_cost = spSolutions[l]->valCost;
        }
        allOpt = (allOpt && isOptimal);
    }

    // Final run of the model
    myData->timeLimit = std::max(1.0, myData->timeLimit); // give at least one second for CPLEX to solve the final model
    myData->regularizationParam = lambdaVector[opt_lambda_index]; // Get optimal regularization parameter
    std::cout << endl << "###  Solving final model with lambda = " << myData->regularizationParam << "  ###" << endl << endl;
    status = finalSolver->solve();
    if (status) {
        std::cerr << "Failed to solve final model" << std::endl;
    }

    // Check optimality of final model run
    allOpt = (allOpt && isSolOpt(finalSolution->solverStatus));

    // Update optimality status
    if (!allOpt) {
        finalSolution->strStatus = "NOT_OPTIMAL";
        finalSolution->solverStatus = -1;
    }

    // Update solution with the final solution
    mySolution->update(finalSolution->numCols, finalSolution->numRows, finalSolution->solverStatus, 
                       finalSolution->strStatus, finalSolution->objVal, finalSolution->lowerBoundVal, 
                       finalSolution->nodeCount, finalSolution->mipGap, &finalSolution->solutionArray[0]);

    // Print solutions found during the search
    std::cout << endl << std::setw(5) << "l" << std::setw(15) << "lambda" << std::setw(15) << "Train cost" << std::setw(15) << "Val cost" << std::setw(15) << "Status" << endl; 
    for (int l=0; l<myData->nbBreakpoints; l++)
        std::cout << std::setw(5) << l << std::setw(15) << lambdaVector[l] << std::setw(15) << spSolutions[l]->trainCost << std::setw(15) << spSolutions[l]->valCost << std::setw(15) << spSolutions[l]->strStatus.c_str() << endl;
    std::cout << endl;

    // Print beta vectors for each l
    std::cout << "Beta vectors:" << endl;
    int width = 12;
    for (int l=0; l<myData->nbBreakpoints; l++) {
        std::cout << "        l = " << std::setw(2) << l;
    }
    std::cout << endl;
    for (int j=0; j<myData->nbFeatures; j++)
    {
        for (int l=0; l<myData->nbBreakpoints; l++) {
            if (isSolOpt(spSolutions[l]->solverStatus)) {
                std::cout << std::setw(width) << spSolutions[l]->solBeta[j];
            } else {
                std::cout << std::setw(width) << "X";
            }
        }
        std::cout << endl;
    }

    // Print optimal solution
    std::cout << endl << "Optimal solution:" << endl;
    std::cout << "Regularization parameter:   " << std::fixed << std::setprecision(5) << myData->regularizationParam << std::endl;
    std::cout << "Training cost:              " << std::fixed << std::setprecision(5) << finalSolution->trainCost << std::endl;
    std::cout << "Validation cost:            " << std::fixed << std::setprecision(5) << finalSolution->valCost << std::endl;
    std::cout << endl;

    myData->timeLimit = origTimeLimit;

    return status;
}

int SolverGridSearchCV::solve()
{
    /*
    For lambda^(l) in the set of breakpoints:
        For k = 1, ..., K
            Solve subproblem_k (lambda^(l)) on training set T_k
            Record solution beta_k^(l)
            Calculate validation cost C_{Vk}^(l)
        Calculate average validation cost \bar{C}_V^(l) = 1/K * sum_{k=1,...,K} C_{Vk}^(l)
    Pick l that minimizes the average validation cost \bar{C}_V^(l)
    */

    int status = 0;
    bool debug = false;

    std::cout << "Solving optimization problem" << endl;

    clock_t time_start = clock(); // Begin of clock
    double time_elapsed = 0;

    double origTimeLimit = myData->timeLimit; // save original time limit

    vector<double> lambdaVector = logspace(-3, 2, myData->nbBreakpoints, 10); // generate breakpoints spaced evenly on a log scale
    lambdaVector.insert(lambdaVector.begin(), 0.0); myData->nbBreakpoints++; // include zero as a breakpoint

    // Initialize Solution and Solver arrays
    std::shared_ptr<SolutionGridSearchCV> mySolutionGS = std::dynamic_pointer_cast<SolutionGridSearchCV>(mySolution);
    vector<vector<std::shared_ptr<SolutionSubProblem>>>& spSolutions = mySolutionGS->spSolutions;

    // Initialize models for each breakpoint and each fold
    for (int k=0; k<myData->nbFolds; k++)
    {
        vector<std::shared_ptr<SolutionSubProblem>> spSolutionsCurrentFold;
        vector<std::unique_ptr<SolverSubProblem>> spSolversCurrentFold;
        for (int l=0; l<myData->nbBreakpoints; l++)
        {
            spSolutionsCurrentFold.push_back(SolutionSubProblem::createSolutionSP(myData));
            spSolversCurrentFold.push_back(SolverSubProblem::createSolverSP(myData, spSolutionsCurrentFold[l]));
            spSolutionsCurrentFold[l]->trainCost = CPX_INFBOUND;
            spSolutionsCurrentFold[l]->valCost   = CPX_INFBOUND;
            spSolutionsCurrentFold[l]->strStatus = "NOT_SOLVED";
        }
        spSolutions.push_back(std::move(spSolutionsCurrentFold));
        spSolvers.push_back(std::move(spSolversCurrentFold));
    }
    for (int l=0; l<myData->nbBreakpoints; l++)
        mySolutionGS->lambdaVector.push_back(lambdaVector[l]);

    int spi = 0; // index of each subproblem
    for (int l=0; l<myData->nbBreakpoints; l++) {
        for (int k=0; k<myData->nbFolds; k++) {
            spSolvers[k][l]->spIndex = spi;
            spi++;
        }
    }

    // Initialize final model
    mySolutionGS->finalSolution = SolutionSubProblem::createSolutionSP(myData);
    std::shared_ptr<SolutionSubProblem> finalSolution = mySolutionGS->finalSolution;
    finalSolver   = SolverSubProblem::createSolverSP(myData, finalSolution);

    int opt_lambda_index = -1; // Optimal number of active features
    double opt_val_cost = CPX_INFBOUND; // Validation cost of optimal solution

    // Record if subproblem solutions are optimal
    bool allOpt = true;

    // Perform cross-validation
    for (int l=0; l<myData->nbBreakpoints; l++) {
        if (debug)
            std::cout << endl << "###  Solving subproblems with lambda = " << mySolutionGS->lambdaVector[l] << "  ###" << endl << endl;
        myData->regularizationParam = lambdaVector[l];  // Set regularization parameter

        bool isOptimal = true;
        double avgValCost = 0; // average validation cost (over all folds)

        vector<double> trainCostVector_aux;
        vector<double> valCostVector_aux;

        for (int k=0; k<myData->nbFolds; k++) {
            if (debug)
                std::cout << "k = " << k << endl;

            // Update hold-out split
            myData->modifySplit(myData->crossValSplit[k]);

            time_elapsed = (((double)(clock() - time_start)) / CLOCKS_PER_SEC);
            
            if (debug)
                std::cout << "Elapsed time: " << time_elapsed << endl;
            if (time_elapsed > origTimeLimit) {
                isOptimal = false;
                allOpt = false;
                break;
            }
            myData->timeLimit = std::max(5e-2, origTimeLimit - time_elapsed); // if time limit was reached, set it to very small value

            status = spSolvers[k][l]->solve(); // Solve subproblem l
            if (status) {
                std::cerr << "Failed to solve subproblem " << l << ", fold " << k << std::endl;
            }

            isOptimal = (isOptimal && isSolOpt(spSolutions[k][l]->solverStatus)); // Check if it is optimal

            avgValCost += spSolutions[k][l]->valCost;

            trainCostVector_aux.push_back(spSolutions[k][l]->trainCost);
            valCostVector_aux.push_back(spSolutions[k][l]->valCost);
        }
        avgValCost /= myData->nbFolds;
        mySolutionGS->avgValCostVector.push_back(avgValCost);

        mySolutionGS->trainCostVector.push_back(std::move(trainCostVector_aux));
        mySolutionGS->valCostVector.push_back(std::move(valCostVector_aux));

        // Update optimal validation cost and the corresponding number of features
        if ((isOptimal == true) && (avgValCost < opt_val_cost - 1e-5)) {
            opt_lambda_index = l;
            opt_val_cost = avgValCost;
        }
        allOpt = (allOpt && isOptimal);
    }

    // Print all values for training and validation costs
    if (myData->nbBreakpoints * myData->nbFolds <= 150)
    {
        // Training cost
        std::cout << "   Train Cost:" << endl;
        std::cout << "            ";
        for (int l=0; l<myData->nbBreakpoints; l++) 
            std::cout << "         l = " << std::setw(2) << std::setfill(' ') << l;
        std::cout << endl;

        for (int k=0; k<myData->nbFolds; k++) {
            std::cout << "   k = " << std::setw(2) << k << "   ";
            for (int l=0; l<myData->nbBreakpoints; l++) {
                std::cout << std::setw(15) << spSolutions[k][l]->trainCost;
            }
            std::cout << endl;
        }

        // Validation cost
        std::cout << endl << endl << "   Val Cost:" << endl;
        std::cout << "            ";
        for (int l=0; l<myData->nbBreakpoints; l++) 
            std::cout << "         l = " << std::setw(2) << l;
        std::cout << endl;

        for (int k=0; k<myData->nbFolds; k++) {
            std::cout << "   k = " << std::setw(2) << k << "   ";
            for (int l=0; l<myData->nbBreakpoints; l++) {
                std::cout << std::setw(15) << spSolutions[k][l]->valCost;
            }
            std::cout << endl;
        }
        std::cout << "  Average   ";
        for (int l=0; l<myData->nbBreakpoints; l++) {
            std::cout << std::setw(15) << mySolutionGS->avgValCostVector[l];
        }
        std::cout << endl << endl;
    }
    
    // Print solutions found during the search
    std::cout << endl << std::setw(5) << "l" << std::setw(15) << "lambda" << std::setw(15) << "Avg val cost" << endl; 
    for (int l=0; l<myData->nbBreakpoints; l++)
        std::cout << std::setw(5) << l << std::setw(15) << lambdaVector[l] << std::setw(15) << mySolutionGS->avgValCostVector[l] << endl;
    std::cout << endl;

    // Final run of the model    
    myData->timeLimit = std::max(1.0, myData->timeLimit); // give at least one second for CPLEX to solve the final model
    myData->regularizationParam = lambdaVector[opt_lambda_index]; // Get optimal regularization parameter
    std::cout << endl << "###  Solving final model with lambda = " << myData->regularizationParam << "  ###" << endl << endl;
    myData->resetSplit(); // Update hold-out split
    status = finalSolver->solve();
    if (status) {
        std::cerr << "Failed to solve final model" << std::endl;
    }

    // Check optimality of final model run
    bool isOptimal = isSolOpt(finalSolution->solverStatus);
    allOpt = (allOpt && isOptimal);

    // Update optimality status
    if (!allOpt) {
        finalSolution->strStatus = "NOT_OPTIMAL";
        finalSolution->solverStatus = -1;
    }

    // Update solution with the final solution
    mySolutionGS->update(finalSolution->numCols, finalSolution->numRows, finalSolution->solverStatus, 
                         finalSolution->strStatus, finalSolution->objVal, finalSolution->lowerBoundVal, 
                         finalSolution->nodeCount, finalSolution->mipGap, &finalSolution->solutionArray[0]);

    // Print optimal solution
    std::cout << endl << "Optimal solution:" << endl;
    std::cout << "Regularization parameter:   " << std::fixed << std::setprecision(5) << myData->regularizationParam << std::endl;
    std::cout << "Training cost:              " << std::fixed << std::setprecision(5) << finalSolution->trainCost << std::endl;
    std::cout << "Validation cost:            " << std::fixed << std::setprecision(5) << finalSolution->valCost << std::endl;
    std::cout << endl;

    myData->timeLimit = origTimeLimit;

    return status;
}
