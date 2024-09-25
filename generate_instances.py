# Code inspired on: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html

import numpy as np
import pandas as pd
import itertools
import os

np.random.seed(0)

dataset_path = "./data/"

### ====================================================================== ###


def generate_instance(n_samples=40, n_test_samples=1000, n_features=10,
                      demand_model='linear', noise_dist='gaussian', noise_level=0,
                      output_path=None):
    
    # Ground-truth beta parameters
    assert n_features > 0
    if n_features >= 4:
        ground_truth = np.concatenate([np.array([2,-2,-1,1]), np.zeros(n_features - 4)]) / np.sqrt(10)
    else:
        ground_truth = np.array([2,-2,-1,1])[:n_features]
    
    # Normalize ground-truth parameter vector
    ground_truth = ground_truth / np.linalg.norm(ground_truth)
    
    # Check that it is unit norm 
    assert np.isclose(np.linalg.norm(ground_truth), 1), "Ground-truth is not unit norm: {}".format(np.linalg.norm(ground_truth))
    
    ground_truth = np.concatenate([np.array([0]), ground_truth]) # add bias term
    ground_truth = ground_truth[:,np.newaxis]
    
    # Check the dimension of the ground-truth vector
    assert ground_truth.shape == (n_features+1, 1)
    
    # Mean vector and covariance matrix of the data
    mean = np.zeros(n_features)
    cov = np.zeros((n_features, n_features))
    for row,col in itertools.product(range(n_features), range(n_features)):
        i = row+1
        j = col+1
        cov[row,col] = 0.5**(np.abs(i-j))
    
    # Samples
    X = np.random.multivariate_normal(mean, cov, size=n_samples)
    X = np.concatenate([np.ones((n_samples,1)), X], axis=1) # add bias column
    
    # Test samples
    X_test = np.random.multivariate_normal(mean, cov, size=n_test_samples)
    X_test = np.concatenate([np.ones((n_test_samples,1)), X_test], axis=1) # add bias column
    
    # Noise
    if noise_dist == 'gaussian': # standard Gaussian N(0,\sigma)
        epsilon = np.random.normal(scale=noise_level, size=(n_samples,1))
        epsilon_test = np.random.normal(scale=noise_level, size=(n_test_samples, 1))
    elif noise_dist == 'mixture': # Gaussian mixture 0.8*N(0,1) + 0.2*N(0,9)
        epsilon = 0.8 * np.random.normal(scale=1.0, size=(n_samples, 1)) + 0.2 * np.random.normal(scale=9.0, size=(n_samples, 1))
        epsilon_test = 0.8 * np.random.normal(scale=1.0, size=(n_test_samples, 1)) + 0.2 * np.random.normal(scale=9.0, size=(n_test_samples, 1))
    elif noise_dist == 'student': # student-t distribution with 3 degrees of freedom
        epsilon = np.random.standard_t(noise_level, size=(n_samples, 1))
        epsilon_test = np.random.standard_t(noise_level, size=(n_test_samples, 1))
    
    # Demand model
    if demand_model == 'linear':
        y = 5 + np.dot(X, ground_truth) + epsilon 
        y_test = 5 + np.dot(X_test, ground_truth) + epsilon_test 
    elif demand_model == 'nlhom':
        y = 10 + np.sin(2*np.dot(X, ground_truth)) + 2*np.exp(-16*np.dot(X, ground_truth)**2) + epsilon
        y_test = 10 + np.sin(2*np.dot(X_test, ground_truth)) + 2*np.exp(-16*np.dot(X_test, ground_truth)**2) + epsilon_test  
    elif demand_model == 'nlhet':
        y = 10 + np.sin(2*np.dot(X, ground_truth)) + 2*np.exp(-16*np.dot(X, ground_truth)**2) + np.exp(np.dot(X, ground_truth))*epsilon
        y_test = 10 + np.sin(2*np.dot(X_test, ground_truth)) + 2*np.exp(-16*np.dot(X_test, ground_truth)**2) + np.exp(np.dot(X_test, ground_truth))*epsilon_test  
    
    assert y.shape == (n_samples, 1)
    assert y_test.shape == (n_test_samples, 1)
    
    # Set negative demand values to zero
    y      = np.maximum(0, y)
    y_test = np.maximum(0, y_test)
    
    if output_path is None:
        return X, y, X_test, y_test, np.squeeze(ground_truth)
    
    # Save instance to output_path
    # Convert instance to Data Frame
    train_out_path = output_path + ".train"
    assert not os.path.exists(train_out_path), "File: {} already exists."\
            .format(os.path.basename(train_out_path))
    df_instance = pd.DataFrame(np.concatenate([y, X], axis=1),
                               columns=["d"]+["x^"+str(j) for j in range(n_features+1)])
    
    # Convert ground truth parameters to data frame
    df_ground_truth = pd.DataFrame(ground_truth).T
    df_ground_truth.columns = ["beta^"+str(j) for j in range(n_features+1)]
    
    # Save ground truth parameters in txt file
    df_ground_truth.to_csv(train_out_path, sep='\t')
    
    with open(train_out_path, "a") as f:
        f.write("\n")
    
    # Save instance to txt file
    df_instance.to_csv(train_out_path, sep='\t', mode='a')
    
    # Save test set
    test_out_path = output_path + ".test"
    assert not os.path.exists(test_out_path)
    df_test = pd.DataFrame(np.concatenate([y_test, X_test], axis=1),
                           columns=["d"]+["x^"+str(j) for j in range(n_features+1)])
    df_test.to_csv(test_out_path, sep='\t')
    

### ====================================================================== ###

counter = 0

### Generate instances 

set_n_samples = [40]+list(range(100,2100,100))
set_n_features = [8,10,12,14]
set_demand_model = ['linear','nlhom','nlhet']
# Default: gaussian with sigma=2 and student with df=3
set_noise_level = [1] 
set_seed = list(range(1,21)) # 1, ... , 20

n_test_samples = 1000

for n_samples, n_features, demand_model, seed in itertools.product(set_n_samples, 
                                                                   set_n_features, 
                                                                   set_demand_model,
                                                                   set_seed):
    for noise_level in set_noise_level:
        # Get path to txt that will be created
        instance_name = "n{}-m{}-{}-l{}-s{}".format(n_samples, n_features, 
                                                    demand_model, noise_level,
                                                    seed)
        out_path = os.path.join(dataset_path, instance_name)
    
        generate_instance(n_samples=n_samples, 
                          n_test_samples=n_test_samples, 
                          n_features=n_features, 
                          demand_model=demand_model, 
                          noise_level=noise_level,
                          output_path=out_path)
        counter += 1
    
print("End of execution > {} instances were generated, {} files created.".format(counter, 2*counter))