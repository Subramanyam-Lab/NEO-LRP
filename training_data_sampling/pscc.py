import math
import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

def plot_customers_and_depots(all_customers, depots, chosen_depot, sampled_customers, output_image_path):
    """
    Plots all customers, depots, the chosen depot, and the sampled customers.
    Saves the plot to the specified path.
    """
    customer_x, customer_y = zip(*[(cust[0], cust[1]) for cust in all_customers])
    depot_x, depot_y = zip(*depots)
    chosen_depot_x, chosen_depot_y = chosen_depot
    if sampled_customers:
        sampled_customers_x, sampled_customers_y = zip(*[(cust[0], cust[1]) for cust in sampled_customers])
    else:
        sampled_customers_x, sampled_customers_y = [], []

    plt.figure(figsize=(10, 8))
    plt.scatter(customer_x, customer_y, color='blue', label='Customers', alpha=0.6)
    plt.scatter(depot_x, depot_y, color='green', marker='s', label='Depots', s=100)
    plt.scatter(chosen_depot_x, chosen_depot_y, color='red', marker='s', label='Chosen Depot', s=200)
    if sampled_customers_x and sampled_customers_y:
        plt.scatter(sampled_customers_x, sampled_customers_y, color='orange', label='Sampled Customers', s=50)

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Customers and Depots Visualization')
    plt.legend()
    plt.savefig(output_image_path, dpi=350)
    plt.close()

def create_data(file_loc):
    """
    Parses the LRP instance file and returns the relevant data.
    """
    data1 = []
    with open(file_loc, 'r') as file:
        for line in file:
            item = line.rstrip().split()
            if len(item) != 0:
                data1.append(item)

    # Number of Customers
    no_cust = int(data1[0][0]) 

    # Number of Depots
    no_depot = int(data1[1][0])

    # Depot coordinates
    depot_cord = []
    ct = 2
    ed_depot = ct + no_depot
    for i in range(ct, ed_depot):
        depot_cord.append((int(data1[i][0]), int(data1[i][1])))

    # Customer coordinates
    cust_cord = []
    ed_cust = ed_depot + no_cust
    for j in range(ed_depot, ed_cust):
        cust_cord.append((int(data1[j][0]), int(data1[j][1])))

    # Vehicle Capacity
    vehicle_cap = int(data1[ed_cust][0])
    vehicle_cap = [vehicle_cap] * no_depot

    # Depot capacities
    depot_cap = []
    start = ed_cust + 1
    end = start + no_depot
    for k in range(start, end):
        depot_cap.append(int(data1[k][0]))

    # Customer Demands
    cust_dem = []
    dem_end = end + no_cust
    for l in range(end, dem_end):
        cust_dem.append(int(data1[l][0]))

    # Opening Cost of Depots
    open_dep_cost = []
    cost_end = dem_end + no_depot
    for x in range(dem_end, cost_end):
        open_dep_cost.append(int(data1[x][0]))

    # Route Cost
    route_cost = int(data1[cost_end][0])

    return [no_cust, no_depot, depot_cord, cust_cord, vehicle_cap, depot_cap, cust_dem, open_dep_cost, route_cost]

def read_lrp_instance(file_loc):
    """
    Reads an LRP instance from the specified file location.
    """
    return create_data(file_loc)

def euclidean_distance(x1, y1, x2, y2):
    """
    Calculates the Euclidean distance between two points.
    """
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def sample_lrp_data_customer_first_optimal_depot(lrp_data, num_customers_to_sample, rng):
    """
    Implements the Customer-First Optimal Depot Assignment under Capacity Constraints.
    Samples customers first, then selects the depot that minimizes total distance.
    """
    all_depots = lrp_data[2]
    all_depot_capacities = lrp_data[5]
    all_customers = [(x, y, demand) for (x, y), demand in zip(lrp_data[3], lrp_data[6])]

    if num_customers_to_sample > len(all_customers):
        logger.warning(f"Requested number of customers ({num_customers_to_sample}) exceeds available customers ({len(all_customers)}).")
        return None

    # Sample N customers uniformly at random
    sampled_customers = rng.sample(all_customers, num_customers_to_sample)

    # Compute total demand
    total_demand = sum(cust[2] for cust in sampled_customers)

    # Identify depots with sufficient capacity
    suitable_depots = []
    for idx, (depot, capacity) in enumerate(zip(all_depots, all_depot_capacities)):
        if capacity >= total_demand:
            suitable_depots.append((idx, depot, capacity))

    if not suitable_depots:
        logger.warning("No depot has sufficient capacity to serve the total demand.")
        return None

    # Compute total distance to customers for each suitable depot
    depot_distances = []
    for idx, depot, capacity in suitable_depots:
        total_distance = sum(
            euclidean_distance(depot[0], depot[1], cust[0], cust[1]) for cust in sampled_customers
        )
        depot_distances.append((total_distance, idx, depot))

    # Select the depot with minimum total distance
    depot_distances.sort(key=lambda x: x[0])
    min_total_distance, chosen_depot_idx, chosen_depot = depot_distances[0]
    logger.debug(f"Selected Depot Index: {chosen_depot_idx}, Total Distance: {min_total_distance}")

    return {
        'sampled_customers': sampled_customers,
        'chosen_depot': chosen_depot,
        'depot_idx': chosen_depot_idx,
        'total_demand': total_demand
    }

def write_to_cvrplib_format(sampled_data, filename, vehicle_capacity):
    """
    Writes the sampled VRP data to a file in CVRPLIB format.
    """
    just_filename = os.path.basename(filename)
    rows = []
    rows.append({'Column1': f'NAME : {just_filename}'})

    rows.append({'Column1': 'COMMENT : Generated by Modified VRP Data Generation'})
    rows.append({'Column1': 'TYPE : CVRP'})
    rows.append({'Column1': f'DIMENSION : {len(sampled_data["sampled_customers"]) + 1}'})

    rows.append({'Column1': 'EDGE_WEIGHT_TYPE : EUC_2D'})
    rows.append({'Column1': f'CAPACITY : {vehicle_capacity}'})
    
    rows.append({'Column1': 'NODE_COORD_SECTION'})

    # Depot is always node 1
    depot = sampled_data['chosen_depot']
    rows.append({'Column1': f"1 {depot[0]} {depot[1]}"})

    for i, point in enumerate(sampled_data['sampled_customers'], start=2):
        rows.append({'Column1': f"{i} {point[0]} {point[1]}"})

    rows.append({'Column1': 'DEMAND_SECTION'})
    
    rows.append({'Column1': "1 0"})  # Depot has zero demand
    
    for i, point in enumerate(sampled_data['sampled_customers'], start=2):
        rows.append({'Column1': f"{i} {point[2]}"})

    rows.append({'Column1': 'DEPOT_SECTION'})
    rows.append({'Column1': "1"})  # Depot ID
    rows.append({'Column1': '-1'})  # EOF for DEPOT_SECTION
    
    rows.append({'Column1': 'EOF'})
    
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False, header=False)

def main(folder_path, seed, output_filename_prefix, num_samples, output_folder, default_instance=None):
    """
    Main function to generate VRP instances based on LRP data using the Customer-First Optimal Depot Assignment.
    """
    # Initialize random seed for reproducibility
    base_seed = seed
    seed_counter = 0

    # Collect all instance files
    all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    if not all_files:
        logger.error("No files found in the specified folder path.")
        return

    logger.info(f"Total LRP instances available: {len(all_files)}")

    instance_idx = 0
    total_instances = len(all_files)

    # Initialize a set to track unique VRP instances
    unique_vrp_instances = set()

    while len(unique_vrp_instances) < num_samples:
        if default_instance:
            selected_file_with_extension = default_instance
            if selected_file_with_extension not in all_files:
                logger.error(f"Default instance '{default_instance}' not found in the folder.")
                return
        else:
            selected_file_with_extension = all_files[instance_idx % total_instances]

        selected_file_without_extension = os.path.splitext(selected_file_with_extension)[0]
        current_seed = base_seed + seed_counter  # Unique seed for each attempt

        logger.info(f"Selected File: {selected_file_with_extension} with Seed: {current_seed}")

        # Read the LRP data
        lrp_data = read_lrp_instance(os.path.join(folder_path, selected_file_with_extension))
        total_customers = lrp_data[0]

        # Initialize a local random generator with the current seed
        rng_main = random.Random(current_seed)

        # Choose N uniformly at random from 1 to total_customers
        num_customers_to_sample = rng_main.randint(1, total_customers)
        logger.debug(f"Number of customers to sample: {num_customers_to_sample}")

        # Sample customers and assign optimal depot
        sampled_data = sample_lrp_data_customer_first_optimal_depot(lrp_data, num_customers_to_sample, rng_main)

        if sampled_data:
            # Create a unique identifier for the VRP instance
            depot_identifier = tuple(sampled_data['chosen_depot'])
            # customer_identifiers = tuple(sorted([(cust[0], cust[1]) for cust in sampled_data['sampled_customers']]))
            customer_identifiers = tuple(sorted([(cust[0], cust[1], cust[2]) for cust in sampled_data['sampled_customers']]))
            vrp_identifier = (depot_identifier, customer_identifiers)

            # Check for duplicates
            if vrp_identifier in unique_vrp_instances:
                logger.info(f"Duplicate VRP instance detected. Continuing to next attempt.")
            else:
                unique_vrp_instances.add(vrp_identifier)
                logger.info(f"Unique VRP instance collected ({len(unique_vrp_instances)}/{num_samples}).")

                # Writing to CVRPLIB format
                output_filename_full = os.path.join(
                    output_folder,
                    f"{output_filename_prefix}_{selected_file_without_extension}_{len(unique_vrp_instances)}_customers_{len(sampled_data['sampled_customers'])}_seed_{current_seed}.txt"
                )
                vehicle_capacity = lrp_data[4][0]

                write_to_cvrplib_format(
                    sampled_data=sampled_data,
                    filename=output_filename_full,
                    vehicle_capacity=vehicle_capacity
                )
        else:
            # Capacity constraints do not hold; skip to next attempt
            logger.warning(f"Capacity constraints not met or no suitable depot found. Skipping to next attempt.")

        # Increment the seed counter to ensure unique seeds for each attempt
        seed_counter += 1

        # Increment instance_idx only if not using default_instance
        if not default_instance:
            instance_idx += 1

    logger.info(f"Sampling completed. Total unique VRP instances collected: {len(unique_vrp_instances)}")

    summary = {
        'Total Unique VRP Instances': len(unique_vrp_instances)
    }

    summary_df = pd.DataFrame([summary])
    summary_output_path = os.path.join(output_folder, f"{output_filename_prefix}_summary.csv")
    summary_df.to_csv(summary_output_path, index=False)
    logger.info(f"Summary report saved to {summary_output_path}")


if __name__ == "__main__":
    main(
        folder_path='NEO-LRP/prodhon_dataset', 
        seed=42, 
        output_filename_prefix='pscc',
        num_samples=10000,
        output_folder='specify_output_folder',
        default_instance=None
    )
