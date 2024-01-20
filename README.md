#  Codes and datasets for the paper Neural Embedded Optimization for Integrated Location and Routing Problems

The repository is structured as follows: 

### 1) `prodhon_dataset` 
This folder contains the dataset used in our numerical experiments.

### 2) `neos` 
Contains scripts for computing LRP costs using neural embedded framework, with results stored in `neos_results.xlsx` under the `results` folder. Key scripts include:
   - `flp_org`: Generates initial solutions for the MIP model.
   - `dataparse`: Converts `.dat` files (prodhon dataset) into a usable format
   - `network`: Transforms pretrained neural network (in ONNX file format) into PyTorch models, invoking `lrp_easy.py` and `solver_cvrp.py`.
   - `lrp_easy`: The lrp_easy.py contains the LRP (Location Routing Problem) model developed using the Gurobi interface. It includes the trained neural network component for predicting the routing cost, which guides the customer assignment decisions. The customer assignments generated are then passed through solver_cvrp.py.This file has all necessary libraries for executing the scripts. Users should install these libraries, step by step how to run the code is describe in subsequent sections.

   - `lrp_easy_random`: The lrp_easy_random.py is same as lrp_easy.py except that in lrp_easy_random.py `rho` neural network is replaced by an random generator that produces output in the same order of magnitude for both cost and the number of routes. 

   - `solver_cvrp`: calls VRPSolverEasy (exact branch price and cut VRP solver) for exact route cost calculation based on customer assignments.

### 3) `pre_trained_model` 
Stores the pretrained neural network in ONNX format for route cost and number of routes prediction.

### 4) `flp` 
Similar to `neos`, this folder contains scripts for parsing data and executing the FLP model. Key files include:
   - `dataparser`: Parses `.dat` files into a usable format
   - `flp_execute`: Main file calling `flp.py`, `vrp.py`, and `solver_cvrp`.
   - `flp.py` : returns the customer assignments to open facilities
   -  `vrp.py`: gives the routing decisions for the above set of assignments and also the costs associated. The same customer assignments are passed through solver_cvrp; this returns the routing costs and decisions using VRPSolverEasy. We have used OR-Tools for solving the FLP model and VRP model; any other solver can also be used.

### Gurobi Installation and Licensing for NEOS-LRP

Before discussing steps to run the code ensure you have Gurobi with a valid license:

1. **Verify Gurobi Installation:** Check if Gurobi is installed on your machine.

2. **Get a Gurobi License:** If needed get a license. Academic users can get a free license at [Gurobi's Academic License page](https://www.gurobi.com/features/academic-named-user-license/). 

### Step-by-Step Instructions for NEOS-LRP

1. **Clone the Repository:**
   - Clone the NEOS-LRP repository to your local machine.

2. **Navigate to the Repository Folder:**
   - Open your terminal or command prompt.
   - Change the current working directory to the cloned NEOS-LRP repository by running: `cd NEOS-LRP-Codes`.

3. **Create a New Conda Environment:**
   - In your terminal, create a new Conda environment named `neos_lrp` with Python version 3.9 using the command:
     ```
     conda create --name neos_lrp python=3.9
     ```

4. **Activate the Environment:**
   - Activate the newly created Conda environment by running:
     ```
     conda activate neos_lrp
     ```

5. **Install Required Libraries:**
   - Install the necessary libraries listed in the `requirements.txt` file by executing:
     ```
     pip install -r requirements.txt
     ```

6. **Update Script Paths:**

   - First create a new folder to store the results call it `results`:
      - Create two `.xlsx` files for example call them  `neos_results.xlsx` and  `flp_results.xlsx`
   
   - In the `neos_vrpeasy.py` script:
     - Update the `directory_path` variable to point to the path of the `prodhon_dataset` folder for example: `NEOS-LRP-Codes/prodhon_dataset`. Use the absolute path.

      - Update the `existing_excel_file` variable to point to the path of the `neos_results.xlxs` file to write the results, for example: `results/neos_results.xlsx`. Use the absolute path.
      
   - In the `lrp_easy` script, change the paths for `phi_loc` and `rho_loc` to specify the absolute paths to `model_phi_new.onnx` and `model_rho_new.onnx`. For example:
       ```
       phi_loc='/Users/yourusername/NEOS-LRP-Codes/pre_trained_model/model_phi_new.onnx'
       rho_loc='/Users/yourusername/NEOS-LRP-Codes/pre_trained_model/model_rho_new.onnx'
       ```
 
7. **Run the `neos_vrpeasy.py` Script:**

   - Determine the version to run: For using the NEOS model with rho neural network predictions, ensure the script begins with `from lrp_easy import createLRP`. If your intention is to test the random generator, the first line should be `from lrp_easy_random import createLRP`

   - Execute the `neos_vrpeasy.py` script located in `NEOS-LRP-Codes/neos`. This runs the neural embedded frameworks, and the results will be stored in `results/neos_results.xlsx`.s


8. **Run the `flp_execute` Script:**
   - Before running `flp_execute`, ensure that you update the `directory_path` variable in the script, similar to the step mentioned in point 6.


9. **Check the `results` folder:**

After successful running the code you can see the two Excel files with results. The files include:
   - `neos_results`: Contains columns for various cost metrics, execution times, and model performance data. The data columns are: File name, FLP cost, NN predicted VRP cost, NN predicted LRP cost, avg lrp_easy script execution time per depot, initial solution generation time, NN model execution time, VRPSolverEasy computed VRP cost, actual LRP cost(using VRPSolverEasy), avg solver_cvrp script execution time per depot, total solver_cvrp script execution time, VRPSolverEasy model solve time.
   - `flp_results`: Presents data related to OR-Tools performance, including costs, solve times, and execution metrics. The data columns are: File name, OR-Tools FLP cost, OR-Tools predicted VRP cost, OR-Tools predicted LRP cost, FLP model solve time, OR-Tools execution total time, OR-Tools avg execution time per depot, VRP cost (VRPSolverEasy), LRP cost (VRPSolverEasy), total solver_cvrp script execution time, avg solver_cvrp script execution time."

### Additional:

## Random Route Costs and Number of Routes in branch random_learning_prediction
 we have introduced a  change to the way route costs and the number of routes are determined in the NEOS LRP model. Instead of using the `rho` neural network to predict these values, we are now generating them randomly within predefined ranges.

## Key Changes:
- **Route Cost Calculation**: Route costs for each depot are now randomly generated within the range of 0.0 to 25.3265 (this range is obtained from training data and is the normalized cost).
- **Number of Routes Estimation**: The number of routes for each depot is randomly chosen between 1 and 19, instead of being predicted by the `rho` neural network.(this range is obtained from training data).
