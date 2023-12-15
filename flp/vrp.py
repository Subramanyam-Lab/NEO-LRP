from ortools.linear_solver import pywraplp
from datetime import datetime
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
from dataparse import create_data, dist_calc
import math
import numpy as np
import logging


class createVRP ():
    def __init__(self,ans,flp_output):
        self.customer_no=ans[0]
        self.depotno=ans[1]
        self.vehicle_capacity=ans[4]  
        self.route_cost= ans[8]  
        self.flp_cost=flp_output[0]
        self.cust_ass=flp_output[1]
        self.cust_dist=flp_output[2]
        self.facility_tot_dem=flp_output[3]
        self.cust_dem_per_fac= flp_output[4]
        self.all_trip_cost=0
        self.no_vehicle=0
        self.routes=[]
        self.fixed_cost=[]
        self.variable_cost=[]
        self.rc_cal_index=ans[9]

    def runVRP (self):
        customer_no=self.customer_no
        for f in self.cust_ass:
            cust_dist_ip =self.cust_dist[f]
            cust_dem_ip = self.cust_dem_per_fac[f]
            veh_cap_ip = self.vehicle_capacity[f]
            agg_dem_ip = self.facility_tot_dem[f]
            final_ans=self.create_data_model(cust_dist_ip,cust_dem_ip,veh_cap_ip,customer_no)
            print("Final Ans:{} for facility {}".format(final_ans,f),'\n\n')
            self.vrp(final_ans)
        route_tot_cost=self.no_vehicle*self.route_cost+self.all_trip_cost
        logging.info(f'The fixed cost for routing {self.no_vehicle*self.route_cost}')
        logging.info(f'Routing Variable cost{self.all_trip_cost}')
        print("Facility Setup cost:",self.flp_cost)
        print("Total cost for routing:",route_tot_cost)
        logging.info(f'Total VRP cost {route_tot_cost}')
        lrp_cost=self.flp_cost+route_tot_cost
        print('Total lrp cost:',(lrp_cost))
        return route_tot_cost, lrp_cost, self.routes,self.fixed_cost,self.variable_cost
        

    def create_data_model(self,all_cord,cust_dem,veh_cap,cust_no):
        """Stores the data for the problem."""
        data = {}
        data['locations'] =all_cord
        data['demands'] = cust_dem
        data['demands'].insert(0,0)
        data['num_vehicles']=cust_no
        l1=[]
        for i in range(data['num_vehicles']):
            l1.append(veh_cap)
        data['vehicle_capacities'] = l1
        data['depot'] = 0
        data['rc_cal_index']=self.rc_cal_index
        return data

    def print_solution(self, data, manager, routing, solution):
        """Prints solution on console."""
        # self.all_trip_cost
        # self.no_vehicle
        num_routes=0
        self.all_trip_cost=self.all_trip_cost+solution.ObjectiveValue()
        self.variable_cost.append(solution.ObjectiveValue())
        print(f'Objective: {solution.ObjectiveValue()}')
        max_route_distance = 0
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
            route_distance = 0
            route_load = 0
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route_load += data["demands"][node_index]
                plan_output += ' {} -> '.format(manager.IndexToNode(index))
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id)
            if route_distance>0:
                plan_output += '{}\n'.format(manager.IndexToNode(index))
                self.routes.append(plan_output)
                plan_output += 'Distance of the route: {}m\n'.format(route_distance)
                plan_output += "Load of the route: {}\n ".format(route_load)
                print(plan_output)
                self.no_vehicle=self.no_vehicle+1
                num_routes+=1
                logging.info(f'Route of vehicle {plan_output}')
                
            max_route_distance = max(route_distance, max_route_distance)
        self.fixed_cost.append(num_routes)
        print('Maximum of the route distances: {}m'.format(max_route_distance))

    def vrp(self,data):

        """Entry point of the program."""
        print("Entered the vrp function")
        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(len(data['locations']),
                                            data['num_vehicles'], data['depot'])

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        dist_matrix=dist_calc(data['locations'],data['locations'],data['rc_cal_index'])
        
        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return dist_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Capacity constraint.
        def demand_callback(from_index):
            """Returns the demand of the node."""
            # Convert from routing variable Index to demands NodeIndex.
            from_node = manager.IndexToNode(from_index)
            return data['demands'][from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(
            demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            data['vehicle_capacities'],  # vehicle maximum capacities
            True,  # start cumul to zero
            'Capacity')
        
        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.FromSeconds(5)

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)
        # Print solution on console.
        if solution:
            print("Final Solution for vrp")
            self.print_solution(data, manager, routing, solution)

        

    

