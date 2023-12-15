
from lrp_easy import createLRP
from solver_cvrp import dataCvrp
import os
import openpyxl
from dataparse import create_data
from datetime import datetime
import logging
import sys


# name_pattern = os.environ.get('name_pattern')
log_dir = "log_files/mip_nn"
os.makedirs(log_dir, exist_ok=True)
# Directory containing the files
directory_path = "/Users/waquarkaleem/NEOS-LRP-Codes-1/prodhon_dataset"

# Iterate through files in the directory
existing_excel_file="/Users/waquarkaleem/NEOS-LRP-Codes-1/results/neos_results.xlsx" 
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
        logging.info(f'\n\n Working on file :{file_path}')
        ans = create_data(file_path)  # Assuming create_data is a function that processes the data
        lrp_st=datetime.now()
        lrp_solver = createLRP(ans)  # Assuming createFlp is a function that creates the solver
        lrp_result=lrp_solver.model(file_path,log_filename)
        print(lrp_result)
        warmstart_time=lrp_result[4] #per depot
        nn_model_time=lrp_result[5]
        logging.info("\n\n")
        logging.info(f"Facility assignment decisions Results{lrp_result}")
        lrp_ed=datetime.now()
        flp_dict={}
        for j in range(len(lrp_result[0])):
            if lrp_result[0][j]>0.5:
                ls=[]
                for i in range(len(lrp_result[1][j])):          
                    if lrp_result[1][j][i]>0.5:
                        ls.append(i)
                
                flp_dict[j]=ls
        rout_dist={}
        fac_cust_dem={}
        cust_dem_fac={}
        for f in flp_dict:
            ls1=[]
            ls2=[]
            dem_sum=0
            for c in (flp_dict[f]):
                ls1.append(ans[3][c])
                dem_sum=dem_sum+ans[6][c]
                ls2.append(ans[6][c])
            ls1.insert(0,ans[2][f])
            ls2.insert(0,0)
            rout_dist[f]=ls1
            fac_cust_dem[f]=dem_sum
            cust_dem_fac[f]=ls2
        ass_result=[lrp_result[2],flp_dict,rout_dist,fac_cust_dem,cust_dem_fac]
        ve_st=datetime.now()
        print("Running vrpeasy function")
        logging.info(f"The Input to vRP easy {ass_result}")
        vrpeasy_solver=dataCvrp(ans,ass_result)
        vrp_easy_results=vrpeasy_solver.runVRPeasy()
        logging.debug("VRP Results",vrp_easy_results)
        ve_ed=datetime.now()

        od=len(flp_dict)
        
        lrp_exec=((lrp_ed-lrp_st).total_seconds())/od
        warmstart_time=warmstart_time/od
        tot_ve_exec=(ve_ed-ve_st).total_seconds()
        ve_exec=((ve_ed-ve_st).total_seconds())/od
        
        # Close the log file
        logging.shutdown()

        # Create a new row
        new_row = [file_path,lrp_result[2],lrp_result[3],lrp_result[2]+lrp_result[3],lrp_exec,warmstart_time,nn_model_time,vrp_easy_results[1], vrp_easy_results[0],ve_exec,tot_ve_exec,vrp_easy_results[7]]  # Replace with your data
        
        # Append the new row to the worksheet
        worksheet.append(new_row)

        # Save the modified Excel file
        workbook.save(existing_excel_file)


