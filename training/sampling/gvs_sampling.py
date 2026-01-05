import os

# Configuration
depot_positions = [1, 2, 3]            # 1=Random, 2=Centered, 3=Cornered
customer_positions = [1, 2, 3]         # 1=Random, 2=Clustered, 3=Random-clustered
demand_types = [1, 2, 3, 4, 5, 6, 7]   # Various demand distributions
avg_route_sizes = [1, 2, 3, 4, 5, 6]   # From very short to ultra long
customer_sizes = list(range(5, 101, 5))  # 5, 10, 15,..., 100

total_instances = 200000  # We sample more here but later use only 120k

configurations_count = (
    len(customer_sizes) * len(depot_positions) *
    len(customer_positions) * len(demand_types) *
    len(avg_route_sizes)
)

instances_per_config = total_instances // configurations_count
extra_instances = total_instances % configurations_count  # remainder to distribute

with open("generation.sh", "w") as bash_script:
    bash_script.write("#!/bin/bash\n\n")
    bash_script.write("# Auto-generated script to run VRP instance generation\n")

    instance_id = 1
    rand_seed = 1

    for n in customer_sizes:
        for depotPos in depot_positions:
            for custPos in customer_positions:
                for demandType in demand_types:
                    for avgRouteSize in avg_route_sizes:
                        reps = instances_per_config
                        if extra_instances > 0:
                            reps += 1
                            extra_instances -= 1

                        for _ in range(reps):
                            command = (
                                f"python generator.py {n} {depotPos} {custPos} "
                                f"{demandType} {avgRouteSize} {instance_id} {rand_seed}\n"
                            )
                            bash_script.write(command)
                            instance_id += 1
                            rand_seed += 1

    bash_script.write("\necho 'All VRP instances have been generated.'\n")

print("Bash script 'run_vrp_generation.sh' has been created successfully.")
