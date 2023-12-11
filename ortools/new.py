from flp import createFlp
from vrp import createVRP
from solver_cvrp import dataCvrp
import os
import openpyxl
from dataparse import create_data
from datetime import datetime
import logging
import sys


name_pattern = os.environ.get('name_pattern')
log_dir = "log_files\ortools"
os.makedirs(log_dir, exist_ok=True)
# Directory containing the files
directory_path = "data"



# Iterate through files in the directory
existing_excel_file="results\ortools_results.xlsx"
workbook = openpyxl.load_workbook(existing_excel_file)

# Select the worksheet where you want to append the new row
worksheet = workbook.active

for filename in os.listdir(directory_path):
    if filename.endswith(".dat"):  # Adjust the file extension as needed
        
        file_path = os.path.join(directory_path, filename)
      
        print("Working on :",file_path)
        # Call your function to get data
        # Create the log file name based on the input file's name
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  
        log_filename = f"{os.path.splitext(filename)[0]}_{current_time}.log"

        logging.basicConfig(
            level=logging.INFO,  # Set the logging level (INFO, WARNING, ERROR, etc.)
            format='%(asctime)s - %(levelname)s - %(message)s',  # Define the log message format
            handlers=[
                logging.FileHandler(os.path.join(log_dir, log_filename)),  # Log to a file
                logging.StreamHandler(sys.stdout)   # Log to stdout
            ]
        )
        ans = create_data(file_path)  # Assuming create_data is a function that processes the data
        flp_solver = createFlp(ans)  # Assuming createFlp is a function that creates the solver
        flp_results = flp_solver.flp()
        logging.debug("FLP Results",flp_results)
        od=len(flp_results[1])
        or_st=datetime.now()
        vrp_solver = createVRP(ans, flp_results)
        or_vrp_results = vrp_solver.runVRP()
        
        logging.debug("VRP Results",or_vrp_results)
        or_ed=datetime.now()
        tot_or_exec=(or_ed-or_st).total_seconds()
        or_exec=(tot_or_exec)/od
        
        ve_st=datetime.now()
        print("Running vrpeasy function")
        vrpeasy_solver=dataCvrp(ans,flp_results)
        vrp_easy_results=vrpeasy_solver.runVRPeasy()
        logging.debug("VRP Results",vrp_easy_results)
        ve_ed=datetime.now()
        tot_ve_exec=(ve_ed-ve_st).total_seconds()
        ve_exec=(tot_ve_exec)/od
        
        # Close the log file
        logging.shutdown()

        # Create a new row
        new_row = [file_path, flp_results[0], or_vrp_results[0],or_vrp_results[1],flp_results[5],tot_or_exec,or_exec,vrp_easy_results[1], vrp_easy_results[0],tot_ve_exec,ve_exec]  # Replace with your data
        
        # Append the new row to the worksheet
        worksheet.append(new_row)

        # Save the modified Excel file
        workbook.save(existing_excel_file)


