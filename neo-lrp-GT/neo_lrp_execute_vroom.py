import os
import shutil
from datetime import datetime
from openpyxl import Workbook
from neural_embedded_model import createLRP
from solver_cvrp_vroom import dataCvrp
from dataparse import create_data

# -----------------------
# Configuration / Paths
# -----------------------
BFS = "solutions"  # BFS folder
phi_loc = "pre_trained_models/graph_transformer/neural_network.pth"  # NN model path
existing_excel_file = "output/results.xlsx"  # Excel results file
sheet_name = "results"
fi_mode_input = "dynamic"

log_dir = "log_files/mip_nn"
os.makedirs(log_dir, exist_ok=True)

# Dataset directory
directory_path = "data/prodhon_dataset"

# Initialize BFS folder
if os.path.exists(BFS):
    print(f"[INFO] Removing old BFS folder: {BFS}")
    shutil.rmtree(BFS)
os.makedirs(BFS, exist_ok=True)
print(f"[INFO] Created fresh BFS folder: {BFS}")

# -----------------------
# Prepare Excel file
# -----------------------
os.makedirs(os.path.dirname(existing_excel_file), exist_ok=True)

try:
    if os.path.exists(existing_excel_file):
        print(f"Existing results file found: {existing_excel_file}. Deleting it.")
        os.remove(existing_excel_file)

    workbook = Workbook()
    workbook.save(existing_excel_file)
    print(f"New results file created: {existing_excel_file}")
except Exception as e:
    print(f"Error while handling Excel file: {e}")
    if os.path.exists(existing_excel_file):
        os.remove(existing_excel_file)
    workbook = Workbook()
    workbook.save(existing_excel_file)

if sheet_name not in workbook.sheetnames:
    workbook.create_sheet(sheet_name)
worksheet = workbook[sheet_name]

# Define headings
headings = [
    "Instance", "FLP", "VRP", "LRP(MIP+NN)", "NumRoutes_OptSol",
    "LA time", "VRPSolverEasy computed VRP cost",
    "actual LRP cost(using VRPSolverEasy)",
    "Solver time", "BKS",
    "Optimization_gap_optsol", "Prediction_gap"
]

# Add headings if sheet is empty
if worksheet.max_row == 1 and worksheet.max_column == 1 and worksheet.cell(1, 1).value is None:
    for col, heading in enumerate(headings, start=1):
        worksheet.cell(row=1, column=col, value=heading)

# -----------------------
# Helper function
# -----------------------
def has_customers(instance_file_path):
    """Check if a VRP instance has any customer assigned."""
    with open(instance_file_path, 'r') as file:
        lines = file.readlines()
    try:
        demand_section_index = lines.index("DEMAND_SECTION\n")
        depot_section_index = lines.index("DEPOT_SECTION\n")
    except ValueError as e:
        raise ValueError(f"Section missing in file {instance_file_path}: {e}")

    customer_demands = lines[demand_section_index + 1:depot_section_index][1:]
    return len(customer_demands) > 0

# -----------------------
# BKS dictionary
# -----------------------
bks_dict = {
    "coord20-5-1.dat": 54793,
    "coord20-5-1b.dat": 39104,
    "coord20-5-2.dat": 48908,
    # ... continue as needed
}

# Track already processed instances
processed_instances = {
    row[0] for row in worksheet.iter_rows(min_row=2, min_col=1, max_col=1, values_only=True) if row[0]
}

# -----------------------
# Main loop over instances
# -----------------------
for filename in bks_dict.keys():
    if filename in processed_instances:
        continue

    file_path = os.path.join(directory_path, filename)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    print(f"Processing instance: {filename}")

    # Initialize lists to collect metrics for averaging
    flp_cost_list = []
    vrp_cost_list = []
    lrp_cost_list = []
    vrp_routes_optsol_list = []
    lrp_exec_list = []
    warmstart_time_list = []
    vrp_easy_vrp_cost_list = []
    actual_lrp_cost_list = []
    ve_exec_list = []
    tot_ve_exec_list = []
    vrp_solver_easy_model_solve_time_list = []
    gap_list = []
    gap_vrp_perc_list = []

    # -----------------------
    # Run multiple times
    # -----------------------
    for run_index in range(5):
        print(f"Run {run_index + 1}")

        # Create subdirectory for this run
        instance_subdir_run = os.path.join(BFS, os.path.splitext(filename)[0], f"run_{run_index}")
        os.makedirs(instance_subdir_run, exist_ok=True)

        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"{os.path.splitext(filename)[0]}_{current_time}_run_{run_index}.log"

        # Load data
        ans = create_data(file_path)

        # Run LRP solver
        lrp_st = datetime.now()
        lrp_solver = createLRP(ans)
        lrp_result = lrp_solver.model(
            file_path,
            log_filename,
            instance_subdir_run,
            phi_loc,
            fi_mode=fi_mode_input,
            fixed_fi_value=1000.0
        )
        lrp_ed = datetime.now()

        warmstart_time = lrp_result[4]
        pred_costs_dict = lrp_result[5]

        # FLP assignment
        flp_dict = {}
        for j, y_val in enumerate(lrp_result[0]):
            if y_val > 0.5:
                flp_dict[j] = [i for i, x in enumerate(lrp_result[1][j]) if x > 0.5]

        # Compute routing distances and demands
        rout_dist = {}
        fac_cust_dem = {}
        cust_dem_fac = {}
        for f in flp_dict:
            coords_list = [ans[3][c] for c in flp_dict[f]]
            dem_list = [ans[6][c] for c in flp_dict[f]]
            dem_sum = sum(dem_list)
            coords_list.insert(0, ans[2][f])
            dem_list.insert(0, 0)
            rout_dist[f] = coords_list
            fac_cust_dem[f] = dem_sum
            cust_dem_fac[f] = dem_list

        ass_result = [lrp_result[2], flp_dict, rout_dist, fac_cust_dem, cust_dem_fac]

        # Run VRP Easy solver
        ve_st = datetime.now()
        vrpeasy_solver = dataCvrp(ans, ass_result)
        vrp_easy_results = vrpeasy_solver.runVRPeasy()
        ve_ed = datetime.now()

        # Metrics computation
        od = len(flp_dict)
        lrp_exec = ((lrp_ed - lrp_st).total_seconds()) / od
        warmstart_time = warmstart_time / od
        tot_ve_exec = (ve_ed - ve_st).total_seconds()
        ve_exec = tot_ve_exec / od

        instance_name = os.path.basename(file_path)
        bks = bks_dict.get(instance_name)

        flp_cost = lrp_result[2]
        vrp_cost = lrp_result[3]
        lrp_cost = flp_cost + vrp_cost
        actual_lrp_cost = vrp_easy_results[0]
        vrp_easy_vrp_cost = vrp_easy_results[1]
        vrp_routes_optsol = sum(vrp_easy_results[3])

        # Compute overall gap
        gap = round(abs(bks - actual_lrp_cost) / bks * 100, 2) if bks else 0

        # Compute per-depot gap
        actual_costs_list = vrp_easy_results[2]
        open_depots = list(flp_dict.keys())
        per_depot_gaps = [
            abs(pred_costs_dict.get(d, 0) - actual_costs_list[k]) / actual_costs_list[k] * 100
            for k, d in enumerate(open_depots) if actual_costs_list[k]
        ]
        gap_vrp_perc = round(sum(per_depot_gaps) / len(per_depot_gaps), 2) if per_depot_gaps else 0

        vrp_solver_easy_model_solve_time = vrp_easy_results[7]

        # Append metrics to lists
        flp_cost_list.append(flp_cost)
        vrp_cost_list.append(vrp_cost)
        lrp_cost_list.append(lrp_cost)
        vrp_routes_optsol_list.append(vrp_routes_optsol)
        lrp_exec_list.append(lrp_exec)
        warmstart_time_list.append(warmstart_time)
        vrp_easy_vrp_cost_list.append(vrp_easy_vrp_cost)
        actual_lrp_cost_list.append(actual_lrp_cost)
        ve_exec_list.append(ve_exec)
        tot_ve_exec_list.append(tot_ve_exec)
        vrp_solver_easy_model_solve_time_list.append(vrp_solver_easy_model_solve_time)
        gap_list.append(gap)
        gap_vrp_perc_list.append(gap_vrp_perc)

    # -----------------------
    # Compute averages
    # -----------------------
    avg_flp_cost = sum(flp_cost_list) / len(flp_cost_list)
    avg_vrp_cost = sum(vrp_cost_list) / len(vrp_cost_list)
    avg_lrp_cost = sum(lrp_cost_list) / len(lrp_cost_list)
    avg_vrp_routes_optsol = sum(vrp_routes_optsol_list) / len(vrp_routes_optsol_list)
    avg_lrp_exec = sum(lrp_exec_list) / len(lrp_exec_list)
    avg_warmstart_time = sum(warmstart_time_list) / len(warmstart_time_list)
    avg_vrp_easy_vrp_cost = sum(vrp_easy_vrp_cost_list) / len(vrp_easy_vrp_cost_list)
    avg_actual_lrp_cost = sum(actual_lrp_cost_list) / len(actual_lrp_cost_list)
    avg_ve_exec = sum(ve_exec_list) / len(ve_exec_list)
    avg_tot_ve_exec = sum(tot_ve_exec_list) / len(tot_ve_exec_list)
    avg_vrp_solver_easy_model_solve_time = sum(vrp_solver_easy_model_solve_time_list) / len(vrp_solver_easy_model_solve_time_list)
    avg_gap = sum(gap_list) / len(gap_list)
    avg_gap_vrp_perc = sum(gap_vrp_perc_list) / len(gap_vrp_perc_list)

    # -----------------------
    # Append row to Excel
    # -----------------------
    new_row = [
        instance_name,
        avg_flp_cost,
        avg_vrp_cost,
        avg_lrp_cost,
        avg_vrp_routes_optsol,
        avg_lrp_exec,
        avg_vrp_easy_vrp_cost,
        avg_actual_lrp_cost,
        avg_tot_ve_exec,
        bks,
        avg_gap,
        avg_gap_vrp_perc
    ]
    worksheet.append(new_row)
    workbook.save(existing_excel_file)
