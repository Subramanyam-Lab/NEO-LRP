# NEOS-LRP
Implementation of Neural Embedded Optimization for Integrated Location and Routing Problems
The GitHub repository has 6 folders:

PS: Please change all the paths to complete file paths for the successful execution of the scripts.

1) requirements.txt: The file requirements.txt lists all the libraries that need to be installed to successfully execute all the scripts.
   
2) data: This folder contains all the data used to obtain the results mentioned in the paper.
   
3) mip+nn: This folder contains all the scripts to calculate LRP costs. The outputs/results generated are written into the neos_results.xlsx file present in the results folder:
  i) The flp_org script is used to generate the initial set of solutions for the MIP model.
  ii) The dataparse script parses the '.dat' files into a usable format for the code.
  iii) The network script converts the ONNX files into PyTorch models. This file calls two other files, lrp_easy.py and solver_cvrp.py:
  iv) The lrp_easy file has the LRP model developed using the Gurobi interface. This includes the trained neural network component for predicting the routing cost, and the neural network guides the customer assignment decisions also. The customer assignments generated are passed through solver_cvrp.
  v) The solver_cvrp calls the VRPSolverEasy, an exact solver which gives the actual route cost for a given set of customer assignments.

4) onnx_files: This folder contains all the trained ONNX files used to predict route cost.

5) flp: This folder has similar files as the mip+nn. The dataparser is used to parse the dat files and convert them into a usable format. The flp_execute is the main file. It calls 3 files: flp.py, which returns the customer assignments to open facilities; vrp.py, which gives the routing decisions for the above set of assignments and also the costs associated. The same customer assignments are passed through solver_cvrp; this returns the routing costs and decisions using VRPSolverEasy. We have used OR-Tools for solving the FLP model and VRP model; any other solver can also be used.

6) results: This folder contains 2 result excel files.
  PS: Please clear the existing data present in the file to generate your set of results for the data.
  > For neos_results, the data columns are: File name, FLP cost, NN predicted VRP cost, NN predicted LRP cost, avg lrp_easy script execution time per depot, initial solution generation time, NN model execution time, VRPeasy computed VRP cost, actual LRP cost(using VRPeasy), avg solver_cvrp script execution time per depot, total solver_cvrp script execution time, VRPeasy model solve time.
  > For flp_results, the data columns are: File name, OR-Tools FLP cost, OR-Tools predicted VRP cost, OR-Tools predicted LRP cost, FLP model solve time, OR-Tools execution total time, OR-Tools avg execution time per depot, VRP cost (VRPeasy), LRP cost (VRPeasy), total solver_cvrp script execution time, avg solver_cvrp script execution time."
