# Bilevel optimization for feature selection in the data-driven newsvendor problem

## Description 

This software performs feature selection for the data-driven newsvendor problem. 

This method is proposed in:
> Breno Serrano, Stefan Minner, Maximilian Schiffer, and Thibaut Vidal (2024). Bilevel optimization for feature selection in the data-driven newsvendor problem. European Journal of Operational Research, 315(2), 703-714. https://doi.org/10.1016/j.ejor.2024.01.025

This repository provides all components required to reproduce the results from the paper, including the data and code for solving the bilevel optimization model and all benchmark methods based on regularization.

## Requirements

The solver is implemented in the programming language C++ and requires CPLEX (version 20.1) to solve the mixed intereger linear programming formulations. 
The code requires Python (version 3.10) for generating the instances. The software was developed and tested on Linux. Compilation uses make and requires g++ to be installed. 

Compilation in Windows is also possible and requires CMake to be installed.

## Installation

- **Linux**: The file `src/Makefile` is provided for compiling the code, which assumes that CPLEX is installed in the directory `/opt/ibm/`. To compile the code, change to `src` directory and run `make`. 

- **Windows**: The files `CMakeLists.txt` and `FindCPLEX.cmake` are provided for compiling the code with CMake. 

## Generation of instances

To create training and testing instances, create a Python virtual environment and install dependencies with `pip install -r requirements.txt`. 
Then, activate the environment and run `python generate_instances.py` to create all instance files in the current working directory. 

## Code Execution

All parameters are passed by CLI arguments to the executable file `newsvendor-features` using:

`./newsvendor-features problem_type instance_path backorder_cost holding_cost [--split train_val_split] [--split_size subset_samples] [--lambda regularization_param] [--folds k] [--breakpts n_bpts] [-t time_limit] [--threads nb_threads] [-o out_path]`

The following CLI arguments are available:

#### Table 1: Description of input arguments

| Argument | Inputs |
| --- | --- |
| problem_type | Method used to perform feature selection (see Table 2 for a description of valid input values) |
| instance_path | Path to instance file |
| backorder_cost | Back-ordering cost $b$ |
| holding_cost | Holding cost $h$ |
| --split | Number of training-validation splits |
| --split_size | Size of training-validation split | 
| --lambda | Regularization parameter $\lambda$ |
| --folds | Number of partitions for cross-validation |
| --breakpts | Number of breakpoints for grid search cross-validation |
| -t | Time limit (in seconds) | 
| --threads | Number of threads to be used by CPLEX | 
| -o | Path to save output file |


#### Table 2: Mapping between codes and methods

| Code | Method |
| --- | --- |
| BL | Bilevel Feature Selection (BFS) |
| BL_SS | BFS with Shuffle & Split cross-validation |
| ERM | Empirical Risk Minimization (without regularization) |
| ERM_l0 | ERM with l0-norm regularization | 
| ERM_l1 | ERM with l1-norm regularization |
| GS_ERM_l0 | ERM_l0 with grid search over regularization parameter values |
| GS_ERM_l1 | ERM_l1 with grid search over regularization parameter values |
| GS_SS_ERM_l0 | GS_ERM_l0 with Shuffle & Split cross-validation |
| GS_SS_ERM_l1 | GS_ERM_l1 with Shuffle & Split cross-validation |

### Example program call

As an example, the following command reads the specified file from the `data` directory and solves the corresponding instance using the Bilevel Feature Selection (BFS) method:

`./newsvendor-features BL ../data/n300-m8-linear-l1-s6.train 1 1`