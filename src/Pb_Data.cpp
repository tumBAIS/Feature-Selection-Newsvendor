#include "Pb_Data.h"

Pb_Data::Pb_Data(string instance_path, string instance_filename, string output_path,
                 double backOrderCost, double holdingCost, ProblemType problemType, string strProblemType, 
                 double trainValSplit, int nominalSplitSize, double regularizationParam,
                 int nbFolds, int nbBreakpoints, int nbThreads, double timeLimit) : 
                 instance_path(instance_path), instance_filename(instance_filename), output_path(output_path),
                 backOrderCost(backOrderCost), holdingCost(holdingCost), 
                 problemType(problemType), strProblemType(strProblemType),
                 trainValSplit(trainValSplit), nominalSplitSize(nominalSplitSize), 
                 regularizationParam(regularizationParam), nbFolds(nbFolds), 
                 nbBreakpoints(nbBreakpoints), nbThreads(nbThreads), timeLimit(timeLimit)
{
    // Get instance name
    int pos = instance_filename.find(this->train_data_extension);
    this->instance_name = instance_filename.substr(0, pos);

    // Get instance base folder
    pos = instance_path.find_last_of("/");
    this->instance_basefolder = this->instance_path.substr(0,pos+1);

    // Get test data path
    this->test_data_filename = this->instance_name + this->test_data_extension;
    this->test_data_path = this->instance_basefolder + this->test_data_filename;

    // Reads training data from the CSV file
    read_data(instance_path.c_str());

    std::cout << "Problem type: " << this->problemType << endl;
    std::cout << "Instance path: " << this->instance_path << endl;
    std::cout << "Test data path: " << this->test_data_path << endl;
    // Reads test data from file
    read_test_data(this->test_data_path);

    // Initializes the beta ground truth vector
    for (int j=0; j<this->nbFeatures; j++)
    {
        double val = std::get<1>(this->ground_truth_params[j])[0];
        ground_truth_beta.push_back(val);
    }

    // Initializes the demand vector
    for (int i=0; i<this->nbSamples; i++)
    {
        double val = std::get<1>(this->dataset[0])[i]; 
        demand.push_back(val);
    }

    // Initialize the feature data matrix
    for (int i=0; i<this->nbSamples; i++)
    {
        std::vector<double> observation;

        for (int j=0; j<this->nbFeatures; j++)
        {
            double val = std::get<1>(this->dataset[j+1])[i]; 
            observation.push_back(val);
        }
        feature_data.push_back(std::move(observation));
    }

    // Initializes the test_demand vector
    for (int i=0; i<this->nbTestSamples; i++)
    {
        double val = std::get<1>(this->test_dataset[0])[i]; 
        test_demand.push_back(val);
    }

    // Initialize the test feature data matrix
    for (int i=0; i<this->nbTestSamples; i++)
    {
        std::vector<double> test_observation;

        for (int j=0; j<this->nbFeatures; j++)
        {
            double val = std::get<1>(this->test_dataset[j+1])[i]; 
            test_observation.push_back(val);
        }
        test_feature_data.push_back(std::move(test_observation));
    }

    // Sets number of informative features (ground truth)
    this->nbInformativeFeatures = 0;
    for (int j=0; j<this->nbFeatures; j++)
    {
        if (ground_truth_beta[j] != 0)
            this->nbInformativeFeatures++;
    }

    // Print data set shape
    std::cout << "Number of samples: " << this->nbSamples << endl;
    std::cout << "Number of features: " << this->nbFeatures << endl;
    std::cout << "Number of informative features: " << this->nbInformativeFeatures << endl;
    std::cout << endl;

    resetSplit();

    // Initialize split size
    this->splitSize = -1;

    // Bilevel model with random permutations cross-validation (Shuffle & Split)
    if ((this->problemType == BL_SS) || 
        (this->problemType == GS_SS_ERM_l0) ||
        (this->problemType == GS_SS_ERM_l1)) {

        // Check if split size is negative or greater than the number of samples
        if ((this->nominalSplitSize <= 0) || (this->nominalSplitSize > this->nbSamples))
            this->splitSize = this->nbSamples;
        else
            this->splitSize = this->nominalSplitSize;

        // Split Training set size
        this->splitTrainSize = int(floor(this->splitSize * this->trainValSplit));
        
        // Split Validation set size
        this->splitValSize = this->splitSize - this->splitTrainSize;

        std::cout << "Number of folds: " << this->nbFolds << endl;
        std::cout << "Nominal split size: " << this->nominalSplitSize << endl;
        std::cout << "Split size: " << this->splitSize << endl;
        std::cout << "Split training size: " << this->splitTrainSize << endl;
        std::cout << "Split validation size: " << this->splitValSize << endl;
        std::cout << "Number of features in each split: " << this->splitNbFeatures << endl;

        // Create a vector that stores the sample ids
        vector<int> idSamples = vector<int>(this->nbSamples);
        for (int i=0; i<this->nbSamples; i++)
            idSamples[i] = i;

        // Split the data
        for (int k=0; k<this->nbFolds; k++)
        {
            // Randomly shuffle sample ids
            std::random_shuffle(idSamples.begin(), idSamples.end());

            // crossValSplit: 
            // 0 if sample is not selected for the current split
            // 1 if sample is in the training set
            // 2 if sample is in the validation set

            // Initialize vector with all zeros
            vector<int> splitVector(this->nbSamples, 0);

            // Split samples into training set - set to 1
            for (int i=0; i<this->splitTrainSize; i++)
                splitVector[idSamples[i]] = 1;

            // Split samples into validation set - set to 2
            for (int i=this->splitTrainSize; i<this->splitSize; i++)
                splitVector[idSamples[i]] = 2;

            this->crossValSplit.push_back(std::move(splitVector));
        }

        // Randomly deactivating some features
        for (int k=0; k<this->nbFolds; k++)
        {
            vector<int> row;
            for (int j=0; j<this->nbFeatures; j++)
            {
                row.push_back(j<this->splitNbFeatures);
                // Randomly shuffle sample ids - always keep the intercept: beta_0
                std::random_shuffle(row.begin()+1, row.end());
            }
            this->activeFeatures.push_back(std::move(row));
        }
    }

    std::cout << "Train-Validation split: " << this->trainValSplit << endl;
    std::cout << "Number of training samples: " << this->nbTrainSamples << endl;
    std::cout << "Number of validation samples: " << this->nbValSamples << endl;
    std::cout << "Number of test samples: " << this->nbTestSamples << endl;
    std::cout << endl;
    
    std::cout << "Back-ordering cost: " << this->backOrderCost << endl;
    std::cout << "Holding cost: " << this->holdingCost << endl;
    std::cout << endl;
}

void Pb_Data::resetSplit()
{
    // Training set size
    this->nbTrainSamples = int(floor(this->nbSamples * this->trainValSplit));

    // Validation set size
    this->nbValSamples = this->nbSamples - this->nbTrainSamples;

    // Hold-out validation split
    this->holdoutSplit.clear();
    for (int i=0; i<this->nbSamples; i++) {
        if (i < this->nbTrainSamples) {
            this->holdoutSplit.push_back(1); // Sample belongs to the training set
        } else {
            this->holdoutSplit.push_back(2); // Sample belongs to the validation set
        }
    }
    assert(this->holdoutSplit.size() == this->nbSamples);
}

void Pb_Data::modifySplit(const vector<int>& newHoldoutSplit)
{
    this->nbTrainSamples = 0;
    this->nbValSamples = 0;

    for (int i=0; i<this->nbSamples; i++) {
        this->holdoutSplit[i] = newHoldoutSplit[i];
        if (this->holdoutSplit[i] == 1)
            this->nbTrainSamples++; 
        else if (this->holdoutSplit[i] == 2)
            this->nbValSamples++;
    }
}

void Pb_Data::read_data(string path)
{
    ifstream file;

	/* parsing the datasets */
	file.open(path);
	if (file.is_open())
	{
        std::string line, colname; // Helper vars
        double val;
        
        // Create a vector of <string, int vector> pairs to store the ground truth parameters
        std::vector<std::pair<std::string, std::vector<double>>> ground_truth_params;

        // Create a vector of <string, int vector> pairs to store the data set
        std::vector<std::pair<std::string, std::vector<double>>> dataset;
        
        // Read the column names
        if(file.good())
        {
            std::getline(file, line); // Extract the first line in the file
            std::stringstream ss(line); // Create a stringstream from line
            std::getline(ss, colname, sep); // Ignore first element (which corresponds to the index)

            // Extract each column name
            while(std::getline(ss, colname, sep)){
                // Initialize and add <colname, int vector> pairs to result
                ground_truth_params.push_back({colname, std::vector<double> {}});
            }
            this->nbFeatures = ground_truth_params.size();

            std::getline(file, line); // Get next line (which contains the data values for the parameter vector)
    
            // Create a stringstream of the current line
            ss.clear();
            ss.str(line);

            int colIdx = 0; // Keep track of the current column index            
            ss >> val; // Ignores first value (which corresponds to the index)
            // Extract each integer
            while(ss >> val){
                ground_truth_params.at(colIdx).second.push_back(val); // Add the current integer to the 'colIdx' column's values vector
                if(ss.peek() == sep) ss.ignore(); // If the next token is a comma, ignore it and move on
                colIdx++; // Increment the column index
            }
            std::getline(file, line); // Get next line (which should contain a line break)

            if (line != "")
            {
				cout << "ERROR when reading instance, wrong format" << endl;
				throw std::runtime_error("ERROR when reading instance, wrong format");
			}

            std::getline(file, line); // Extract line containing the column names

            // Create a stringstream from line
            ss.clear();
            ss.str(line);

            std::getline(ss, colname, sep); // Ignore first element

            // Extract each column name
            while(std::getline(ss, colname, sep)){
                // Initialize and add <colname, int vector> pairs to result
                dataset.push_back({colname, std::vector<double> {}});
            }
            if (this->nbFeatures + 1 != dataset.size())
            {
				cout << "ERROR when reading instance, inconsistent number of features" << endl;
				throw std::runtime_error("ERROR when reading instance, inconsistent number of features");
			}

            int nbSamples = 0;
            // Read data, line by line
            while(std::getline(file, line))
            {
                // Create a stringstream of the current line
                ss.clear();
                ss.str(line);                
                
                int colIdx = 0; // Keep track of the current column index
                ss >> val; // Ignores first value (which corresponds to the index)
                
                // Extract each integer
                while(ss >> val){
                    
                    // Add the current integer to the 'colIdx' column's values vector
                    dataset.at(colIdx).second.push_back(val);
                    
                    // If the next token is a comma, ignore it and move on
                    if(ss.peek() == sep) ss.ignore();
                    
                    // Increment the column index
                    colIdx++;
                }
                nbSamples++;
            }
            this->nbSamples = nbSamples;
        }
        file.close();

        this->ground_truth_params = ground_truth_params;
        this->dataset = dataset;
    }
}


void Pb_Data::read_test_data(string path)
{
    ifstream file;

	/* parsing the datasets */
	file.open(path);
	if (file.is_open())
	{
        std::string line, colname; // Helper vars
        double val;

        // Create a vector of <string, int vector> pairs to store the data set
        std::vector<std::pair<std::string, std::vector<double>>> test_dataset;
        
        // Read the column names
        if(file.good())
        {
            std::getline(file, line); // Extract the first line in the file
            std::stringstream ss(line); // Create a stringstream from line
            std::getline(ss, colname, sep); // Ignore first element (which corresponds to the index)

            // Extract each column name
            while(std::getline(ss, colname, sep)){
                // Initialize and add <colname, int vector> pairs to result
                test_dataset.push_back({colname, std::vector<double> {}});
            }
            if (this->nbFeatures + 1 != test_dataset.size())
            {
				cout << "ERROR when reading test data, inconsistent number of features" << endl;
				throw std::runtime_error("ERROR when reading test data, inconsistent number of features");
			}

            int nbTestSamples = 0;
            // Read data, line by line
            while(std::getline(file, line))
            {
                // Create a stringstream of the current line
                ss.clear();
                ss.str(line);                
                
                int colIdx = 0; // Keep track of the current column index
                ss >> val; // Ignores first value (which corresponds to the index)
                
                // Extract each integer
                while(ss >> val){
                    
                    // Add the current integer to the 'colIdx' column's values vector
                    test_dataset.at(colIdx).second.push_back(val);
                    
                    // If the next token is a comma, ignore it and move on
                    if(ss.peek() == sep) ss.ignore();
                    
                    // Increment the column index
                    colIdx++;
                }
                nbTestSamples++;
            }
            this->nbTestSamples = nbTestSamples;
        }
        file.close();

        this->test_dataset = test_dataset;
    }
}


void Pb_Data::display_table(std::vector<std::pair<std::string, std::vector<double>>> table)
{
    int nbFeatures = table.size(); // Gets number of features
    int nbSamples = std::get<1>(table[0]).size(); // Gets number of samples

    for (int j=0; j<nbFeatures; j++) 
    {
        std::cout << std::get<0>(table[j]) << '\t'; // Prints column names
    }
    std::cout << endl;

    for (int i=0; i<nbSamples; i++)
    {
        for (int j=0; j<nbFeatures; j++) 
        {
            std::cout << std::get<1>(table[j])[i] << '\t'; // Prints values
        }
        std::cout << endl;
    }
}
