#include "Solution.h"

bool isSolOpt(int solver_status)
{
    return ((solver_status == CPX_STAT_OPTIMAL) || 
            (solver_status == CPXMIP_OPTIMAL) ||
            (solver_status == CPXMIP_OPTIMAL_TOL));
}

//------------------------------------------------------------------------------

void SolutionSubProblem_ERM_l0::updateSolutionVectors()
{
    int sizeU, sizeO, sizeBeta, sizeZ;
    sizeU = sizeO = myData->nbTrainSamples;
    sizeBeta = sizeZ = myData->nbFeatures;
    for (int i=0; i<sizeU; i++)
    {
        this->solU.push_back(solutionArray[i]);
        this->solO.push_back(solutionArray[sizeU+i]);
    }

    for (int j=0; j<sizeBeta; j++)
    {
        solBeta.push_back(solutionArray[sizeU + sizeO + j]);
        solZ.push_back(solutionArray[sizeU + sizeO + sizeBeta + j]);
    }
}

void SolutionSubProblem_ERM_l1::updateSolutionVectors()
{
    int sizeU, sizeO, sizeBeta;
    sizeU = sizeO = myData->nbTrainSamples;
    sizeBeta = myData->nbFeatures;
    for (int i=0; i<sizeU; i++)
    {
        solU.push_back(solutionArray[i]);
        solO.push_back(solutionArray[sizeU+i]);
    }

    for (int j=0; j<sizeBeta; j++)
    {
        solBeta_pos.push_back(solutionArray[sizeU + sizeO + j]);
        solBeta_neg.push_back(solutionArray[sizeU + sizeO + sizeBeta + j]);

        solBeta.push_back(solBeta_pos[j] - solBeta_neg[j]);

        if ((solBeta_pos[j] < 1e-6) && (solBeta_neg[j] < 1e-6))
            solZ.push_back(0);
        else 
            solZ.push_back(1);
    }
}

void SolutionGridSearch_ERM_l0::updateSolutionVectors()
{
    sizeU = sizeO = myData->nbTrainSamples;
    sizeBeta = sizeZ = myData->nbFeatures;
    for (int i=0; i<sizeU; i++)
    {
        solU.push_back(solutionArray[i]);
        solO.push_back(solutionArray[sizeU+i]);
    }

    for (int j=0; j<sizeBeta; j++)
    {
        solBeta.push_back(solutionArray[sizeU + sizeO + j]);
        solZ.push_back(solutionArray[sizeU + sizeO + sizeBeta + j]);
    }

    for (int l=0; l<myData->nbBreakpoints; l++)
    {
        trainCostVector.push_back(spSolutions[l]->trainCost);
        valCostVector.push_back(spSolutions[l]->valCost);
    }
}

void SolutionGridSearch_ERM_l1::updateSolutionVectors()
{
    sizeU = sizeO = myData->nbTrainSamples;
    sizeBeta = myData->nbFeatures;
    sizeSoln = sizeU + sizeO + sizeBeta*2;
    for (int i=0; i<sizeU; i++)
    {
        solU.push_back(solutionArray[i]);
        solO.push_back(solutionArray[sizeU+i]);
    }

    for (int j=0; j<sizeBeta; j++)
    {
        solBeta_pos.push_back(solutionArray[sizeU + sizeO + j]);
        solBeta_neg.push_back(solutionArray[sizeU + sizeO + sizeBeta + j]);
        solBeta.push_back(solBeta_pos[j] - solBeta_neg[j]);
    }

    for (int l=0; l<myData->nbBreakpoints; l++)
    {
        trainCostVector.push_back(spSolutions[l]->trainCost);
        valCostVector.push_back(spSolutions[l]->valCost);
    }
}

void SolutionGridSearchCV_ERM_l0::updateSolutionVectors()
{
    sizeU = sizeO = myData->nbTrainSamples;
    sizeBeta = sizeZ = myData->nbFeatures;
    for (int i=0; i<sizeU; i++)
    {
        solU.push_back(solutionArray[i]);
        solO.push_back(solutionArray[sizeU+i]);
    }

    for (int j=0; j<sizeBeta; j++)
    {
        solBeta.push_back(solutionArray[sizeU + sizeO + j]);
        solZ.push_back(solutionArray[sizeU + sizeO + sizeBeta + j]);
    }
}

void SolutionGridSearchCV_ERM_l1::updateSolutionVectors()
{
    sizeU = sizeO = myData->nbTrainSamples;
    sizeBeta = myData->nbFeatures;
    for (int i=0; i<sizeU; i++)
    {
        solU.push_back(solutionArray[i]);
        solO.push_back(solutionArray[sizeU+i]);
    }

    for (int j=0; j<sizeBeta; j++)
    {
        solBeta_pos.push_back(solutionArray[sizeU + sizeO + j]);
        solBeta_neg.push_back(solutionArray[sizeU + sizeO + sizeBeta + j]);
        solBeta.push_back(solBeta_pos[j] - solBeta_neg[j]);
    }
}

// ----------------------------------------------------------------

void SolutionSubProblem_ERM_l0::exportSolutionArrays(std::ofstream &myFile) 
{
    myFile << "regularization_param" << myData->sep << myData->regularizationParam << endl;
    
    if (myData->save_full_soln) {
        exportVector(solU, myFile, "sol_u", myData->sep);           // Export solution u
        exportVector(solO, myFile, "sol_o", myData->sep);           // Export solution o
    }
    exportVector(solBeta, myFile, "sol_beta", myData->sep);     // Export solution beta
    exportVector(solZ, myFile, "sol_z", myData->sep);           // Export solution z
    exportVector(myData->ground_truth_beta, myFile, "ground_truth", myData->sep);
}

void SolutionSubProblem_ERM_l1::exportSolutionArrays(std::ofstream &myFile) 
{
    myFile << "regularization_param" << myData->sep << myData->regularizationParam << endl;
    
    if (myData->save_full_soln) {
        exportVector(solU, myFile, "sol_u", myData->sep);                // Export solution u
        exportVector(solO, myFile, "sol_o", myData->sep);                // Export solution o
    }
    exportVector(solBeta_pos, myFile, "sol_beta_pos", myData->sep);  // Export solution beta_pos
    exportVector(solBeta_neg, myFile, "sol_beta_neg", myData->sep);  // Export solution beta_neg
    exportVector(solBeta, myFile, "sol_beta", myData->sep);          // Export solution beta
    exportVector(myData->ground_truth_beta, myFile, "ground_truth", myData->sep);
}

void SolutionGridSearch_ERM_l0::exportSolutionArrays(std::ofstream &myFile) 
{
    myFile << "regularization_param" << myData->sep << myData->regularizationParam << endl;
    myFile << "num_breakpoints" << myData->sep << myData->nbBreakpoints << endl;
    
    if (myData->save_full_soln) {
        exportVector(solU, myFile, "sol_u", myData->sep);           // Export solution u
        exportVector(solO, myFile, "sol_o", myData->sep);           // Export solution o
    }
    exportVector(solBeta, myFile, "sol_beta", myData->sep);     // Export solution beta
    exportVector(solZ, myFile, "sol_z", myData->sep);           // Export solution z
    exportVector(myData->ground_truth_beta, myFile, "ground_truth", myData->sep);   // Export ground truth parameters
    exportVector(lambdaVector, myFile, "lambda_values", myData->sep); // Export break-points for the regularization parameter
    if (myData->save_full_soln) {
        exportVector(trainCostVector, myFile, "train_cost_values", myData->sep);
        exportVector(valCostVector, myFile, "val_cost_values", myData->sep);
    }
}

void SolutionGridSearch_ERM_l1::exportSolutionArrays(std::ofstream &myFile)
{
    myFile << "regularization_param" << myData->sep << myData->regularizationParam << endl;
    myFile << "num_breakpoints" << myData->sep << myData->nbBreakpoints << endl;
    
    if (myData->save_full_soln) {
        exportVector(solU, myFile, "sol_u", myData->sep);           // Export solution u
        exportVector(solO, myFile, "sol_o", myData->sep);           // Export solution o
    }
    exportVector(solBeta_pos, myFile, "sol_beta_pos", myData->sep);  // Export solution beta_pos
    exportVector(solBeta_neg, myFile, "sol_beta_neg", myData->sep);  // Export solution beta_neg
    exportVector(solBeta, myFile, "sol_beta", myData->sep);          // Export solution beta
    exportVector(myData->ground_truth_beta, myFile, "ground_truth", myData->sep);   // Export ground truth parameters
    exportVector(lambdaVector, myFile, "lambda_values", myData->sep); // Export break-points for the regularization parameter
    if (myData->save_full_soln) {
        exportVector(trainCostVector, myFile, "train_cost_values", myData->sep);
        exportVector(valCostVector, myFile, "val_cost_values", myData->sep);
    }
}

void SolutionGridSearchCV_ERM_l0::exportSolutionArrays(std::ofstream &myFile)
{
    myFile << "nb_folds" << myData->sep << myData->nbFolds << endl;
    myFile << "nominal_split_size" << myData->sep << myData->nominalSplitSize << endl;
    myFile << "split_size" << myData->sep << myData->splitSize << endl;
    myFile << "split_train_size" << myData->sep << myData->splitTrainSize << endl;
    myFile << "split_val_size" << myData->sep << myData->splitValSize << endl;
    
    myFile << "regularization_param" << myData->sep << myData->regularizationParam << endl;
    myFile << "num_breakpoints" << myData->sep << myData->nbBreakpoints << endl;

    if (myData->save_full_soln) {
        exportVector(solU, myFile, "sol_u", myData->sep);           // Export solution u
        exportVector(solO, myFile, "sol_o", myData->sep);           // Export solution o
    }
    exportVector(solBeta, myFile, "sol_beta", myData->sep);     // Export solution beta
    exportVector(solZ, myFile, "sol_z", myData->sep);           // Export solution z
    exportVector(myData->ground_truth_beta, myFile, "ground_truth", myData->sep);   // Export ground truth parameters
    exportVector(lambdaVector, myFile, "lambda_values", myData->sep); // Export break-points for the regularization parameter
    if (myData->save_full_soln) {
        exportMatrix(trainCostVector, myFile, "train_cost_values", myData->sep);
        exportMatrix(valCostVector, myFile, "val_cost_values", myData->sep);
    }
}


void SolutionGridSearchCV_ERM_l1::exportSolutionArrays(std::ofstream &myFile)
{
    myFile << "nb_folds" << myData->sep << myData->nbFolds << endl;
    myFile << "nominal_split_size" << myData->sep << myData->nominalSplitSize << endl;
    myFile << "split_size" << myData->sep << myData->splitSize << endl;
    myFile << "split_train_size" << myData->sep << myData->splitTrainSize << endl;
    myFile << "split_val_size" << myData->sep << myData->splitValSize << endl;
    
    myFile << "regularization_param" << myData->sep << myData->regularizationParam << endl;
    myFile << "num_breakpoints" << myData->sep << myData->nbBreakpoints << endl;
    
    if (myData->save_full_soln) {
        exportVector(solU, myFile, "sol_u", myData->sep);           // Export solution u
        exportVector(solO, myFile, "sol_o", myData->sep);           // Export solution o
    }
    exportVector(solBeta_pos, myFile, "sol_beta_pos", myData->sep);  // Export solution beta_pos
    exportVector(solBeta_neg, myFile, "sol_beta_neg", myData->sep);  // Export solution beta_neg
    exportVector(solBeta, myFile, "sol_beta", myData->sep);          // Export solution beta
    exportVector(myData->ground_truth_beta, myFile, "ground_truth", myData->sep);   // Export ground truth parameters
    exportVector(lambdaVector, myFile, "lambda_values", myData->sep); // Export break-points for the regularization parameter
    if (myData->save_full_soln) {
        exportMatrix(trainCostVector, myFile, "train_cost_values", myData->sep);
        exportMatrix(valCostVector, myFile, "val_cost_values", myData->sep);
    }
}