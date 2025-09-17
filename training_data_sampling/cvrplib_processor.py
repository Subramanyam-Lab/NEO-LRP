"""
CVRPLIB Data Processor for neo-lrp-GT Training Pipeline

This module provides functionality to process CVRPLIB instances (.vrp format)
and convert them to HDF5 format suitable for training the Graph Transformer network.

The pipeline:
1. Reads .vrp files from a directory using vrplib
2. Generates solutions using VROOM if .sol files don't exist
3. Converts the data to HDF5 format compatible with train.py

Usage:
    processor = CVRPLIBProcessor(input_folder, output_file)
    processor.process_all()
"""

import time
import logging
import shutil
import signal
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import h5py
import numpy as np
import vrplib
import vroom
from scipy.spatial import distance_matrix

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    """Exception raised when VROOM solver exceeds time limit."""
    pass


def timeout_wrapper(func, timeout_duration, *args, **kwargs):
    """
    Cross-platform timeout wrapper for function execution.

    Args:
        func: Function to execute
        timeout_duration: Timeout in seconds
        *args, **kwargs: Arguments for the function

    Returns:
        Function result or raises TimeoutError
    """
    result = [None]
    exception = [None]

    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout_duration)

    if thread.is_alive():
        # Thread is still running, timeout occurred
        logger.warning(f"VROOM solver timed out after {timeout_duration} seconds")
        # Note: We cannot forcefully terminate the thread, but we can return early
        raise TimeoutError(f"VROOM solver exceeded time limit of {timeout_duration} seconds")

    if exception[0]:
        raise exception[0]

    return result[0]


class CVRPLIBProcessor:
    """
    Processes CVRPLIB instances and converts them to HDF5 format for neural network training.
    """

    def __init__(self, input_folder: str, output_file: str, force_regenerate: bool = False, time_limit: int = 300):
        """
        Initialize the CVRPLIB processor.

        Args:
            input_folder: Path to folder containing .vrp files
            output_file: Output HDF5 file path
            force_regenerate: If True, regenerate solutions even if .sol files exist
            time_limit: Time limit for VROOM solver in seconds (default: 300)
        """
        self.input_folder = Path(input_folder)
        self.output_file = Path(output_file)
        self.force_regenerate = force_regenerate
        self.time_limit = time_limit
        self.processed_count = 0
        self.failed_count = 0

        # Ensure output directory exists
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

    def find_vrp_files(self) -> List[Path]:
        """
        Find all .vrp files in the input folder.

        Returns:
            List of Path objects for .vrp files
        """
        vrp_files = list(self.input_folder.glob("*.vrp"))
        logger.info(f"Found {len(vrp_files)} .vrp files in {self.input_folder}")
        return vrp_files

    def read_vrp_instance(self, vrp_file: Path) -> Optional[Dict]:
        """
        Read a CVRPLIB instance using vrplib.

        Args:
            vrp_file: Path to .vrp file

        Returns:
            Dictionary containing instance data or None if failed
        """
        try:
            instance = vrplib.read_instance(str(vrp_file))
            logger.debug(f"Successfully read {vrp_file.name}")
            return instance
        except Exception as e:
            logger.error(f"Failed to read {vrp_file.name}: {e}")
            return None

    def read_solution(self, sol_file: Path) -> Optional[Dict]:
        """
        Read a solution file if it exists.

        Args:
            sol_file: Path to .sol file

        Returns:
            Dictionary containing solution data or None if file doesn't exist or failed
        """
        if not sol_file.exists():
            return None

        try:
            solution = vrplib.read_solution(str(sol_file))
            logger.debug(f"Successfully read solution {sol_file.name}")
            return solution
        except Exception as e:
            logger.warning(f"Failed to read solution {sol_file.name}: {e}")
            return None

    def backup_solution_file(self, sol_file: Path) -> Path:
        """
        Create a backup of an existing solution file with timestamp.

        Args:
            sol_file: Path to existing .sol file

        Returns:
            Path to the backup file
        """
        if not sol_file.exists():
            raise FileNotFoundError(f"Solution file {sol_file} does not exist")

        # Create backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{sol_file.stem}_backup_{timestamp}.sol"
        backup_file = sol_file.parent / backup_name

        # Copy the file
        shutil.copy2(sol_file, backup_file)
        logger.info(f"Backed up {sol_file.name} to {backup_file.name}")

        return backup_file

    def generate_vroom_solution(self, instance: Dict) -> Optional[Tuple[Dict, float]]:
        """
        Generate solution using VROOM solver.

        Args:
            instance: CVRPLIB instance dictionary

        Returns:
            Tuple of (solution_dict, solve_time) or None if failed
        """
        try:
            # Extract instance data
            coords = instance['node_coord']
            demands = instance['demand']
            capacity = instance['capacity']
            depot_idx = instance['depot'][0]  # vrplib already converts to 0-based indexing

            # Validate depot index
            if depot_idx < 0 or depot_idx >= len(coords):
                logger.error(f"Invalid depot index: {depot_idx}, coords length: {len(coords)}")
                return None

            # Build distance matrix
            coord_array = np.array(coords)
            dist_matrix = distance_matrix(coord_array, coord_array)

            # Scale distances to integers (VROOM requirement)
            if instance.get('edge_weight_type') == 'EUC_2D':
                dist_matrix = (dist_matrix * 100).astype(int)
            else:
                dist_matrix = dist_matrix.astype(int)

            # Create VROOM problem
            problem = vroom.Input()
            problem.set_durations_matrix("car", dist_matrix.tolist())

            # Add vehicles (generous allocation - one per customer)
            num_customers = len(coords) - 1  # Exclude depot
            for v_id in range(num_customers):
                vehicle = vroom.Vehicle(
                    id=v_id + 1,
                    start=depot_idx,
                    end=depot_idx,
                    profile="car",
                    capacity=[capacity],
                    costs=vroom.VehicleCosts(fixed=1000)
                )
                problem.add_vehicle(vehicle)

            # Add jobs (customers)
            for i, demand in enumerate(demands):
                if i == depot_idx:  # Skip depot
                    continue
                job = vroom.Job(
                    id=i + 1000,  # Unique job ID
                    location=i,
                    delivery=[demand]
                )
                problem.add_job(job)

            # Solve with timeout
            start_time = time.time()
            try:
                solution = timeout_wrapper(
                    problem.solve,
                    self.time_limit,
                    exploration_level=5,
                    nb_threads=4
                )
                solve_time = time.time() - start_time
                logger.debug(f"VROOM solved successfully in {solve_time:.2f}s (limit: {self.time_limit}s)")
            except TimeoutError as e:
                logger.error(f"VROOM solver timeout: {e}")
                return None
            except Exception as e:
                logger.error(f"VROOM solver error: {e}")
                return None

            # Convert VROOM solution to standard format
            routes = []
            for _, group in solution.routes.groupby("vehicle_id"):
                route = group.sort_values("arrival")["location_index"].tolist()
                routes.append(route)

            solution_dict = {
                'routes': routes,
                'cost': solution.summary.cost,
                'solve_time': solve_time
            }

            logger.debug(f"VROOM solved with cost {solution.summary.cost} in {solve_time:.2f}s")
            return solution_dict, solve_time

        except Exception as e:
            logger.error(f"VROOM solution failed: {e}")
            return None

    def save_solution(self, solution: Dict, sol_file: Path, solve_time: float):
        """
        Save solution to .sol file with timing information.

        Args:
            solution: Solution dictionary
            sol_file: Path to save solution
            solve_time: Time taken to solve
        """
        try:
            # Write solution in standard VRP format manually
            # since vrplib.write_solution has issues
            with open(sol_file, 'w') as f:
                # Write routes
                for i, route in enumerate(solution['routes'], 1):
                    route_str = ' '.join(map(str, route))
                    f.write(f"Route #{i}: {route_str}\n")

                # Write cost
                f.write(f"Cost {solution['cost']}\n")

                # Write timing information
                f.write(f"# VROOM solve time: {solve_time:.4f} seconds\n")

            logger.debug(f"Saved solution to {sol_file.name}")
        except Exception as e:
            logger.warning(f"Failed to save solution {sol_file.name}: {e}")

    def convert_to_hdf5_format(self, instance: Dict, solution: Dict) -> Optional[Dict]:
        """
        Convert CVRPLIB instance and solution to HDF5 format compatible with train.py.

        Args:
            instance: CVRPLIB instance dictionary
            solution: Solution dictionary

        Returns:
            Dictionary in HDF5 format or None if conversion failed
        """
        try:
            # Extract coordinates and demands
            coords = np.array(instance['node_coord'])
            demands = np.array(instance['demand'])
            depot_idx = instance['depot'][0]  # vrplib already converts to 0-based indexing

            # Create node features: [x, y, is_depot, demand]
            x_coordinates = coords[:, 0].astype(np.float32)
            y_coordinates = coords[:, 1].astype(np.float32)
            is_depot = np.zeros(len(coords), dtype=np.float32)
            is_depot[depot_idx] = 1.0
            demands = demands.astype(np.float32)

            # Calculate distance matrix
            dist_matrix = distance_matrix(coords, coords).astype(np.float32)

            # Extract cost from solution
            cost = float(solution['cost'])

            # Create mask (all nodes are valid for this format)
            mask = np.ones(len(coords), dtype=np.float32)

            hdf5_data = {
                'x_coordinates': x_coordinates,
                'y_coordinates': y_coordinates,
                'demands': demands,
                'is_depot': is_depot,
                'dist_matrix': dist_matrix,
                'cost': cost,
                'mask': mask
            }

            return hdf5_data

        except Exception as e:
            logger.error(f"Failed to convert to HDF5 format: {e}")
            return None

    def process_single_instance(self, vrp_file: Path) -> bool:
        """
        Process a single CVRPLIB instance.

        Args:
            vrp_file: Path to .vrp file

        Returns:
            True if successfully processed, False otherwise
        """
        logger.info(f"Processing {vrp_file.name}")

        # Read instance
        instance = self.read_vrp_instance(vrp_file)
        if instance is None:
            return False

        # Check for existing solution
        sol_file = vrp_file.with_suffix('.sol')
        solution = self.read_solution(sol_file)
        solve_time = 0.0

        # Determine if we need to generate a new solution
        need_regeneration = solution is None or self.force_regenerate

        if need_regeneration:
            # If force regenerating and solution exists, backup the original
            if self.force_regenerate and solution is not None:
                logger.info(f"Force regenerating solution for {vrp_file.name}, backing up original")
                self.backup_solution_file(sol_file)

            # Generate new solution
            if self.force_regenerate and solution is not None:
                logger.info(f"Force regenerating solution for {vrp_file.name} with VROOM")
            else:
                logger.info(f"No solution found for {vrp_file.name}, generating with VROOM")
            result = self.generate_vroom_solution(instance)
            if result is None:
                return False
            solution, solve_time = result

            # Save generated solution
            self.save_solution(solution, sol_file, solve_time)
        else:
            # Extract solve time if available
            solve_time = solution.get('solve_time', 0.0)
            logger.debug(f"Using existing solution for {vrp_file.name}")

        # Convert to HDF5 format
        hdf5_data = self.convert_to_hdf5_format(instance, solution)
        if hdf5_data is None:
            return False

        # Save to HDF5 file
        instance_name = vrp_file.stem
        try:
            with h5py.File(self.output_file, 'a') as f:
                if instance_name in f:
                    del f[instance_name]  # Remove existing entry

                grp = f.create_group(instance_name)
                for key, value in hdf5_data.items():
                    grp.create_dataset(key, data=value)

                # Add metadata
                grp.attrs['instance_file'] = str(vrp_file.name)
                grp.attrs['solve_time'] = solve_time
                grp.attrs['num_nodes'] = len(hdf5_data['x_coordinates'])
                grp.attrs['num_customers'] = np.sum(1 - hdf5_data['is_depot'])

            logger.info(f"Successfully processed {vrp_file.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to save to HDF5: {e}")
            return False

    def process_all(self):
        """
        Process all .vrp files in the input folder.
        """
        vrp_files = self.find_vrp_files()

        if not vrp_files:
            logger.warning("No .vrp files found!")
            return

        logger.info(f"Starting processing of {len(vrp_files)} instances")
        start_time = time.time()

        for vrp_file in vrp_files:
            if self.process_single_instance(vrp_file):
                self.processed_count += 1
            else:
                self.failed_count += 1

        total_time = time.time() - start_time

        logger.info(f"Processing complete!")
        logger.info(f"Successfully processed: {self.processed_count}")
        logger.info(f"Failed: {self.failed_count}")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Output saved to: {self.output_file}")


def main():
    """
    Example usage of CVRPLIBProcessor.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Process CVRPLIB instances for neo-lrp-GT training")
    parser.add_argument("input_folder", help="Folder containing .vrp files")
    parser.add_argument("output_file", help="Output HDF5 file path")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--force-regenerate", action="store_true",
                       help="Force regenerate solutions even if .sol files exist (original files will be backed up)")
    parser.add_argument("--time-limit", type=int, default=300,
                       help="Time limit for VROOM solver in seconds (default: 300)")

    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Process instances
    processor = CVRPLIBProcessor(args.input_folder, args.output_file, args.force_regenerate, args.time_limit)
    processor.process_all()


if __name__ == "__main__":
    main()