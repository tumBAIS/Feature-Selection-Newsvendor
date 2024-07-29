#include "Solution.h"

// ----------------------------------------------------------------------------------------------------

void SolutionERM::updateSolutionVectors()
{
    int sizeU, sizeO, sizeBeta;
    sizeU = sizeO = myData->nbSamples;
    sizeBeta = myData->nbFeatures;
    for (int i=0; i<sizeU; i++)
    {
        solU.push_back(solutionArray[i]);
        solO.push_back(solutionArray[sizeU+i]);
    }

    for (int j=0; j<sizeBeta; j++)
    {
        solBeta.push_back(solutionArray[sizeU + sizeO + j]);
    }
}

void SolutionERM_l0::updateSolutionVectors()
{
    int sizeU, sizeO, sizeBeta, sizeZ;
    sizeU = sizeO = myData->nbSamples;
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


void SolutionERM_l1::updateSolutionVectors()
{
    int sizeU, sizeO, sizeBeta;
    sizeU = sizeO = myData->nbSamples;
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

void SolutionERM::exportSolutionArrays(std::ofstream &myFile) 
{
    if (myData->save_full_soln) {
        exportVector(solU, myFile, "sol_u", myData->sep);         // Export solution u
        exportVector(solO, myFile, "sol_o", myData->sep);         // Export solution o
    }
    exportVector(solBeta, myFile, "sol_beta", myData->sep);   // Export solution beta
    exportVector(myData->ground_truth_beta, myFile, "ground_truth", myData->sep);   // Export ground truth parameters
}

void SolutionERM_l0::exportSolutionArrays(std::ofstream &myFile) 
{
    myFile << "regularization_param" << myData->sep << myData->regularizationParam << endl;
    
    if (myData->save_full_soln) {
        exportVector(solU, myFile, "sol_u", myData->sep);           // Export solution u
        exportVector(solO, myFile, "sol_o", myData->sep);           // Export solution o
    }
    exportVector(solBeta, myFile, "sol_beta", myData->sep);     // Export solution beta
    exportVector(solZ, myFile, "sol_z", myData->sep);           // Export solution z
    exportVector(myData->ground_truth_beta, myFile, "ground_truth", myData->sep);   // Export ground truth parameters
}

void SolutionERM_l1::exportSolutionArrays(std::ofstream &myFile) 
{
    myFile << "regularization_param" << myData->sep << myData->regularizationParam << endl;
    
    if (myData->save_full_soln) {
        exportVector(solU, myFile, "sol_u", myData->sep);                // Export solution u
        exportVector(solO, myFile, "sol_o", myData->sep);                // Export solution o
    }
    exportVector(solBeta_pos, myFile, "sol_beta_pos", myData->sep);  // Export solution beta_pos
    exportVector(solBeta_neg, myFile, "sol_beta_neg", myData->sep);  // Export solution beta_neg
    exportVector(solBeta, myFile, "sol_beta", myData->sep);          // Export solution beta
    exportVector(myData->ground_truth_beta, myFile, "ground_truth", myData->sep);   // Export ground truth parameters
}

