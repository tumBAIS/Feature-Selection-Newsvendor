#include "Solution.h"

template<typename T>
void exportVector(const vector<T>& vec, std::ofstream& file, string varname, char sep)
{
    file << varname << sep << "[";
    for (int i=0; i<vec.size(); i++)
    {
        if (i!=0)
            file << " ";
        file << vec[i];
    }
    file << "]" << endl;
}

template<typename T>
void exportMatrix(const vector<vector<T>>& mat, std::ofstream& file, string varname, char sep)
{
    // Export solution beta
    file << varname << sep << "[";
    for (int i=0; i<mat.size(); i++)
    {
        file << "[";
        for (int j=0; j<mat[0].size(); j++)
        {
            if (j!=0)
                file << " ";
            file << mat[i][j];
        }
        file << "]";
        if (i < mat.size()-1)
            file << ",";
    }
    file << "]" << endl;
}

// Constructor
Solution::Solution(std::shared_ptr<Pb_Data> myData) : myData(myData)
{
    solverStatus = -1; // not solved yet
}

std::shared_ptr<Solution> Solution::createSolution(std::shared_ptr<Pb_Data> myData)
{
    if (myData->problemType == BL)
        return std::make_shared<SolutionBilevel> (myData);
    
    else if (myData->problemType == BL_SS)
        return std::make_shared<SolutionBilevelShuffleSplit> (myData);
    
    else if (myData->problemType == ERM)
        return std::make_shared<SolutionERM> (myData);

    else if (myData->problemType == ERM_l0)
        return std::make_shared<SolutionERM_l0> (myData);
    
    else if (myData->problemType == ERM_l1)
        return std::make_shared<SolutionERM_l1> (myData);

    else if (myData->problemType == SP_ERM_l0)
        return std::make_shared<SolutionSubProblem_ERM_l0> (myData);

    else if (myData->problemType == SP_ERM_l1)
        return std::make_shared<SolutionSubProblem_ERM_l1> (myData);

    else if (myData->problemType == GS_ERM_l0) // Grid search for ERM-l0
        return std::make_shared<SolutionGridSearch_ERM_l0> (myData);

    else if (myData->problemType == GS_ERM_l1) // Grid search for ERM-l1
        return std::make_shared<SolutionGridSearch_ERM_l1> (myData);

    else if (myData->problemType == GS_SS_ERM_l0)
        return std::make_shared<SolutionGridSearchCV_ERM_l0> (myData);

    else if (myData->problemType == GS_SS_ERM_l1)
        return std::make_shared<SolutionGridSearchCV_ERM_l1> (myData);
    
    else
    {   
        fprintf(stderr, "Failed to create solution structure: no solution matching the input could be found \n");
        return NULL;
    }
}


std::shared_ptr<SolutionSubProblem> SolutionSubProblem::createSolutionSP(std::shared_ptr<Pb_Data> myData)
{
    if ((myData->problemType == GS_ERM_l0) || // Grid search for ERM-l0
        (myData->problemType == GS_SS_ERM_l0)) // Grid search Shuffle & Split for ERM-l0
        return std::make_shared<SolutionSubProblem_ERM_l0> (myData);

    else if ((myData->problemType == GS_ERM_l1) || // Grid search for ERM-l1
             (myData->problemType == GS_SS_ERM_l1)) // Grid search Shuffle & Split for ERM-l1
        return std::make_shared<SolutionSubProblem_ERM_l1> (myData);

    else
    {
        fprintf(stderr, "Failed to create solution structure for subproblem: no solution matching the input could be found \n");
        return NULL;
    }
}


string Solution::doubleToString(double inputVal)
{
    // Only works with precision 2!
    std::stringstream stream;
    stream << std::fixed << std::setprecision(2) << inputVal;
    string auxString = stream.str();

    if (auxString.back() == '0')
        auxString = auxString.substr(0, auxString.length()-1);

        if (auxString.back() == '0')
            auxString = auxString.substr(0, auxString.length()-2);

    string resultString;
    std::remove_copy(auxString.begin(), auxString.end(), std::back_inserter(resultString), '.');
    return resultString;
}


string Solution::getOutputFilename(string extension)
{
    // Generate output filename
    string out_filename = myData->strProblemType + "-" + myData->instance_name + \
                                "-b" + doubleToString(myData->backOrderCost) + \
                                "-h" + doubleToString(myData->holdingCost) + \
                                "-t" + to_string((int) (100*myData->trainValSplit));
    if (myData->nbFolds > 0) {
        out_filename = out_filename + "-fd" + to_string((int) myData->nbFolds);
    }
    if (myData->splitSize > 0) {
        out_filename = out_filename + "-sz" + to_string((int) myData->splitSize);
    }
    if (myData->nbBreakpoints > 0) {
        out_filename = out_filename + "-bp" + to_string((int) myData->nbBreakpoints);
    }
    string out_path = myData->output_path + out_filename + extension; 
    return out_path;
}

void Solution::copyInputFile()
{
    // Create copy of input file
    string infile_cpy = myData->output_path + myData->instance_name + ".in";
    std::ifstream src(myData->instance_path, std::ios::binary);
    std::ofstream dst(infile_cpy,   std::ios::binary);
    dst << src.rdbuf();
}

double Solution::calculateTrainingCost(const vector<double>& beta)
{
    double u, o;
    // Calculate cost on training set
    double cost = 0;
    for (int i=0; i<myData->nbSamples; i++)
    {
        if (myData->holdoutSplit[i] == 1) {
            u = myData->demand[i];
            o = - myData->demand[i];
            for (int j=0; j<myData->nbFeatures; j++)
            {
                u -= beta[j] * myData->feature_data[i][j];
                o += beta[j] * myData->feature_data[i][j];
            }
            u = max(0.0, u);
            o = max(0.0, o);
            cost += myData->backOrderCost*u + myData->holdingCost*o;
        }
    }
    cost /= myData->nbTrainSamples;
    return cost;
}

double Solution::calculateValCost(const vector<double>& beta)
{
    double u, o;
    // Calculate cost on validation set
    double cost = 0;
    for (int i=0; i<myData->nbSamples; i++)
    {
        if (myData->holdoutSplit[i] == 2) {
            u = myData->demand[i];
            o = - myData->demand[i];
            for (int j=0; j<myData->nbFeatures; j++)
            {
                u -= beta[j] * myData->feature_data[i][j];
                o += beta[j] * myData->feature_data[i][j];
            }
            u = max(0.0, u);
            o = max(0.0, o);
            cost += myData->backOrderCost*u + myData->holdingCost*o;
        }
    }
    cost /= myData->nbValSamples;
    return cost;
}


double Solution::calculateTrainValCost(const vector<double>& beta)
{
    double u, o;
    // Calculate cost on training + validation set
    double cost = 0;
    for (int i=0; i<myData->nbSamples; i++)
    {
        u = myData->demand[i];
        o = - myData->demand[i];
        for (int j=0; j<myData->nbFeatures; j++)
        {
            u -= beta[j] * myData->feature_data[i][j];
            o += beta[j] * myData->feature_data[i][j];
        }
        u = max(0.0, u);
        o = max(0.0, o);
        cost += myData->backOrderCost*u + myData->holdingCost*o;
    }
    cost /= myData->nbSamples;
    return cost;
}

double Solution::calculateTestCost(const vector<double>& beta)
{
    double u, o;
    // Calculate cost on test set
    double cost = 0;
    for (int i=0; i<myData->nbTestSamples; i++)
    {
        u = myData->test_demand[i];
        o = - myData->test_demand[i];
        for (int j=0; j<myData->nbFeatures; j++)
        {
            u -= beta[j] * myData->test_feature_data[i][j];
            o += beta[j] * myData->test_feature_data[i][j];
        }
        u = max(0.0, u);
        o = max(0.0, o);
        cost += myData->backOrderCost*u + myData->holdingCost*o;
    }
    cost /= myData->nbTestSamples;
    return cost;
}


// -------------------------------------------------------------------------------------------

void Solution::update(int numCols, int numRows, int solverStatus, string strStatus, double objVal, 
                      double lowerBoundVal, int nodeCount, double mipGap, double *solution)
{
    this->numCols = numCols;
    this->numRows = numRows;
    this->solverStatus = solverStatus;
    this->strStatus = strStatus;
    this->objVal = objVal;
    this->lowerBoundVal = lowerBoundVal;
    this->nodeCount = nodeCount;
    this->mipGap = mipGap;
    assert(this->solutionArray.size() == 0);
    for (int c=0; c<numCols; c++)
        this->solutionArray.push_back(solution[c]);
    assert(this->solutionArray.size() == numCols);

    // Update solution vectors
    updateSolutionVectors();

    // Calculate cost on training set
    this->trainCost = calculateTrainingCost(solBeta);

    // Calculate cost on validation set
    this->valCost = calculateValCost(solBeta);

    // Calculate cost on training + validation set
    this->trainValCost = calculateTrainValCost(solBeta);

    // Calculate cost on test set
    this->testCost = calculateTestCost(solBeta);
}

// -------------------------------------------------------------------------------------------

// ====================================================
// Bilevel model - update
// ====================================================

void SolutionBilevel::updateSolutionVectors()
{
    sizeU = sizeO = myData->nbSamples;
    sizeBeta = sizeZ = myData->nbFeatures;
    sizeMu = sizeGamma = myData->nbTrainSamples;
    sizeSoln = sizeU + sizeO + sizeBeta + sizeMu + sizeGamma + sizeZ;
    for (int i=0; i<sizeU; i++)
    {
        solU.push_back(solutionArray[i]);
        solO.push_back(solutionArray[sizeU+i]);
    }

    for (int j=0; j<sizeBeta; j++)
    {
        solBeta.push_back(solutionArray[sizeU + sizeO + j]);
        solZ.push_back(solutionArray[sizeU + sizeO + sizeBeta + sizeMu + sizeGamma + j]);
    }

    for (int i=0; i<sizeMu; i++)
    {
        solMu.push_back(solutionArray[sizeU + sizeO + sizeBeta + i]);
        solGamma.push_back(solutionArray[sizeU + sizeO + sizeBeta + sizeMu + i]);
    }
}


// ====================================================
// Bilevel model with random permutations cross-validation a.k.a. Shuffle & Split - update
// ====================================================

void SolutionBilevelShuffleSplit::updateSolutionVectors()
{
    int sizeU, sizeO, sizeBeta, sizeMu, sizeGamma, sizeZ;
    sizeU = sizeO = myData->nbFolds * myData->splitSize;
    sizeBeta = myData->nbFolds * myData->nbFeatures;
    sizeMu = sizeGamma = myData->nbFolds * myData->splitTrainSize;
    sizeZ = myData->nbFeatures;

    // Solution variable u
    int r = 0;
    for (int k=0; k<myData->nbFolds; k++)
    {
        vector<double> row;
        for (int i=0; i<myData->nbSamples; i++)
        {
            // consider only samples selected in the split
            if (myData->crossValSplit[k][i] > 0) 
            {
                row.push_back(solutionArray[r]);
                r++;
            }
        }
        solU.push_back(std::move(row));
    }

    // Solution variable o
    r = 0;
    for (int k=0; k<myData->nbFolds; k++)
    {
        vector<double> row;
        for (int i=0; i<myData->nbSamples; i++)
        {
            // consider only samples selected in the split
            if (myData->crossValSplit[k][i] > 0)
            {
                row.push_back(solutionArray[sizeU + r]);
                r++;
            }
        }
        solO.push_back(std::move(row));
    }
    
    // Solution variable beta
    for (int k=0; k<myData->nbFolds; k++)
    {
        vector<double> row;
        for (int j=0; j<myData->nbFeatures; j++)
        {
            row.push_back(solutionArray[sizeU + sizeO + k*myData->nbFeatures + j]);
        }
        solBetaMatrix.push_back(std::move(row));
    }

    // Solution variable mu and gamma
    for (int r=0; r<sizeMu; r++)
    {
        solMu.push_back(solutionArray[sizeU + sizeO + sizeBeta + r]);
        solGamma.push_back(solutionArray[sizeU + sizeO + sizeBeta + sizeMu + r]);
    }

    // Solution variable z
    for (int r=0; r<sizeZ; r++)
    {
        solZ.push_back(solutionArray[sizeU + sizeO + sizeBeta + sizeMu + sizeGamma + r]);
    }

    // Average beta vector
    for (int j=0; j<myData->nbFeatures; j++)
    {
        double avg_val = 0;
        for (int k=0; k<myData->nbFolds; k++)
        {
            avg_val += solBetaMatrix[k][j];
        }
        avg_val /= myData->nbFolds;
        solBeta.push_back(avg_val);
    }
}

// ====================================================
// Export solution
// ====================================================

void Solution::exportSolution(string ext)
{
    string path = getOutputFilename(ext);

    std::cout << "Exporting solution to file: " << path << endl;

    // Create an output filestream object
    std::ofstream myFile(path);

    if (myFile.is_open())
    {
        // Send data to the stream
        myFile << myData->sep << "Solution" << endl;

        // Input parameters
        myFile << "instance_path" << myData->sep << myData->instance_path << endl;
        myFile << "instance_filename" << myData->sep << myData->instance_filename << endl;
        myFile << "instance_name" << myData->sep << myData->instance_name << endl;
        myFile << "problem_type" << myData->sep << myData->strProblemType << endl;
        myFile << "nb_samples" << myData->sep << myData->nbSamples << endl;
        myFile << "nb_features" << myData->sep << myData->nbFeatures << endl;
        myFile << "nb_informative_features" << myData->sep << myData->nbInformativeFeatures << endl;
        myFile << "informative_factor" << myData->sep << myData->informativeFactor << endl;
        myFile << "train_val_split" << myData->sep << myData->trainValSplit << endl;
        myFile << "nb_train_samples" << myData->sep << myData->nbTrainSamples << endl;
        myFile << "nb_val_samples" << myData->sep << myData->nbValSamples << endl;
        myFile << "nb_test_samples" << myData->sep << myData->nbTestSamples << endl;
        myFile << "backorder_cost" << myData->sep << myData->backOrderCost << endl;
        myFile << "holding_cost" << myData->sep << myData->holdingCost << endl;
        myFile << "time_limit" << myData->sep << myData->timeLimit << endl;
        myFile << "nb_threads" << myData->sep << myData->nbThreads << endl;

        // Model parameters
        myFile << "num_cols" << myData->sep << numCols << endl;
        myFile << "num_rows" << myData->sep << numRows << endl;
        myFile << "status_solver" << myData->sep << solverStatus << endl;
        myFile << "status_string" << myData->sep << strStatus << endl;
        myFile << std::fixed << std::setprecision(10); // set precision
        myFile << "solve_time" << myData->sep << solution_time << endl;
        myFile << "obj_val" << myData->sep << objVal << endl;
        
        // MIP parameters
        myFile << "lower_bound" << myData->sep << lowerBoundVal << endl;
        myFile << "node_count" << myData->sep << nodeCount << endl;
        myFile << "mip_gap" << myData->sep << mipGap << endl;

        // Cost performance
        myFile << "train_cost" << myData->sep << trainCost << endl;
        myFile << "val_cost" << myData->sep << valCost << endl;
        myFile << "train_val_cost" << myData->sep << trainValCost << endl;
        myFile << "test_cost" << myData->sep << testCost << endl;

        myFile << std::setprecision(5); // set precision
        this->exportSolutionArrays(myFile);
    
        // Close the file
        myFile.close();
    } else {
        fprintf(stderr, "ERROR: Unable to open file \n");
    }
}

void SolutionBilevel::exportSolutionArrays(std::ofstream &myFile) 
{
    if (myData->save_full_soln) {
        exportVector(solU, myFile, "sol_u", myData->sep);                               // Export solution u
        exportVector(solO, myFile, "sol_o", myData->sep);                               // Export solution o
    }
    exportVector(solBeta, myFile, "sol_beta", myData->sep);                         // Export solution beta
    if (myData->save_full_soln) {
        exportVector(solMu, myFile, "sol_mu", myData->sep);                             // Export solution mu
        exportVector(solGamma, myFile, "sol_gamma", myData->sep);                       // Export solution gamma
    }
    exportVector(solZ, myFile, "sol_z", myData->sep);                               // Export solution z
    exportVector(myData->ground_truth_beta, myFile, "ground_truth", myData->sep);   // Export ground truth parameters
}

void SolutionBilevelShuffleSplit::exportSolutionArrays(std::ofstream &myFile) 
{
    // Information about cross-validation
    myFile << "nb_folds" << myData->sep << myData->nbFolds << endl;
    myFile << "nominal_split_size" << myData->sep << myData->nominalSplitSize << endl;
    myFile << "split_size" << myData->sep << myData->splitSize << endl;
    myFile << "split_train_size" << myData->sep << myData->splitTrainSize << endl;
    myFile << "split_val_size" << myData->sep << myData->splitValSize << endl;
    myFile << "split_features" << myData->sep << myData->splitFeatures << endl;
    myFile << "split_nb_features" << myData->sep << myData->splitNbFeatures << endl;

    exportMatrix(myData->activeFeatures, myFile, "active_features", myData->sep);   // Export active features
    if (myData->save_full_soln) {
        exportMatrix(solU, myFile, "sol_u", myData->sep);                               // Export solution u
        exportMatrix(solO, myFile, "sol_o", myData->sep);                               // Export solution o
    }
    exportMatrix(solBetaMatrix, myFile, "sol_beta_raw", myData->sep);                   // Export solution beta
    exportVector(solBeta, myFile, "sol_beta", myData->sep);                     // Export average solution beta
    if (myData->save_full_soln) {
        exportVector(solMu, myFile, "sol_mu", myData->sep);                             // Export solution mu
        exportVector(solGamma, myFile, "sol_gamma", myData->sep);                       // Export solution gamma
    }
    exportVector(solZ, myFile, "sol_z", myData->sep);                               // Export solution z
    exportVector(myData->ground_truth_beta, myFile, "ground_truth", myData->sep);   // Export ground truth parameters
}

// ====================================================
// Display solution
// ====================================================


void Solution::displaySolution()
{
    // Print results
    printf("\n");
    printf("Problem type:                        %s \n", myData->strProblemType.c_str());
    printf("Solution status:                     %d (%s) \n", solverStatus, strStatus.c_str());
    printf("Solution time:                       %0.10f \n", solution_time);
    printf("Objective value:                     %0.10f \n", objVal);
    printf("Objective lower bound:               %0.10f \n", lowerBoundVal);
    printf("Relative objective gap (MIP):        %0.10f \n", mipGap);
    printf("Nodes processed:                     %d \n", nodeCount);
    std::cout << endl;
    printf("Cost on training set:                 %0.10f \n", trainCost);
    printf("Cost on validation set:               %0.10f \n", valCost);
    printf("Cost on training + validation set:    %0.10f \n", trainValCost);
    printf("Cost on test set:                     %0.10f \n", testCost);
    std::cout << endl;   
}
