""" This module allows to solve CVRPLIB instances of
Capacitated Vehicle Routing Problem """
import math
import getopt
import sys
import os
import argparse
import time
from dataparse import create_data
from VRPSolverEasy.src import solver
import logging
import os
import sys
from datetime import datetime


class dataCvrp ():
    """Contains all data for CVRP problem
    """

    def __init__(self,ans,flp_ass):
        
        self.nb_customers = ans[0]
        self.nb_open_depot=list(flp_ass[1].keys())
        self.coord=flp_ass[2]
        self.vehicle_capacity = ans[4]
        self.rc_cal_index=ans[9]
        self.flp_cost=flp_ass[0]
        self.vrp_cost=0
        self.tot_routes=0
        self.flp_ass=flp_ass
        self.message=[]
        self.variable_cost=[]
        self.num_routes=[]
        self.actual_routes=[]
        self.solve_time=[]
        
        

    def runVRPeasy(self):
        for j in self.nb_open_depot:
            ass_cord=self.coord[j]
            cust_demands = self.flp_ass[4][j]
            cust=self.flp_ass[1][j]
            vrp_data=self.create_data_model(ass_cord,self.nb_open_depot,cust_demands,self.vehicle_capacity,self.nb_customers,cust,self.rc_cal_index)
            cost,m,message,routes,exec_time=self.solve_files_in_directory(vrp_data,cust)
            self.actual_routes.append(routes)
            self.variable_cost.append(cost)
            self.num_routes.append(m)
            self.solve_time.append(exec_time)
            self.message.append(message)
            self.vrp_cost+=cost
            print(cost,m)
            self.tot_routes+=m
            print("sequence of routes:",routes)
            logging.info(f"Sequence of routes is {routes}")
        print("Final VRP variable cost:",self.vrp_cost)
        logging.info(f"Final VRP variable cost is {self.vrp_cost}")
        print("Number of routes:",self.tot_routes)
        logging.info(f"Number of routes is {self.tot_routes}")
        print("Final VRP fixed cost:",1000*self.tot_routes)
        logging.info(f"Final VRP fixed cost is {1000*self.tot_routes}")
        total_lrp_cost=self.flp_cost+self.vrp_cost+1000*self.tot_routes
        total_vrp_cost=self.vrp_cost+1000*self.tot_routes
        print("Total LRP cost",total_lrp_cost)
        logging.info(f"Final VRP cost is {total_lrp_cost}")
        return total_lrp_cost,total_vrp_cost,self.variable_cost,self.num_routes,self.message,self.solve_time,self.actual_routes


    def compute_euclidean_distance(self,x_i, y_i, x_j, y_j,rc_cal_index):
        """Compute the euclidean distance between 2 points from graph"""
        if rc_cal_index==0:
            return int(100*(math.hypot(((x_i - x_j)),((y_i - y_j)))))
        else:
            return int(math.hypot(((x_i - x_j)),((y_i - y_j))))


    def create_data_model(self,all_cord,op_dep,cust_dem,veh_cap,customer_no,cust,rc_cal_index):
        """Stores the data for the problem."""
        data = {}
        data['locations'] =all_cord
        data['open_depot']=op_dep
        data["nb_cust"]=customer_no
        data['demands'] = cust_dem
        #data['demands'].insert(0,0)
        data['vehicle_capacities'] = veh_cap
        data['depot'] = 0
        data['cust_cor']=all_cord[1:]
        data['depot_cor']=all_cord[0]
        data['assigned_cust']=cust
        data['rc_cal_index']=rc_cal_index
        return data

    def solve_demo(self,data,cust,
                time_resolution=7200,
                solver_name_input="CLP",
                solver_path=""):
        """return a solution from modelisation"""

        # modelisation of problem
        model = solver.Model()

        print(data['nb_cust'],data['vehicle_capacities'][0])
        logging.info(f"Number of Max customers {data['nb_cust']}")
        logging.info(f"Vehicle Capacity {data['vehicle_capacities'][0]}")
        # add vehicle type
        model.add_vehicle_type(id=1,
                            start_point_id=0,
                            end_point_id=0,
                            max_number=data['nb_cust'],
                            capacity=data['vehicle_capacities'][0],
                            var_cost_dist=1 # we dont have any variable cost for distance. It is incorporated from the euclidean distance only?
                            )
        # add depot
        model.add_depot(id=0)

        # add all customers
        for i in range(1,len(data["locations"])):
            print(i,data["demands"][i])

            model.add_customer(id=i,
                            demand=data["demands"][i]                           
                            )

        
        # Compute the links between depot and other points
        for i in range(1,len(data['locations'])):
            print(data['locations'][i],data["locations"][0])
            dist = self.compute_euclidean_distance(float(data['locations'][i][0]),
                                            float(data['locations'][i][1]),
                                            float(data['locations'][0][0]),
                                            float(data['locations'][0][1]),
                                            data['rc_cal_index']
                                            )
            print("Dist b/w  cust {} and depot is  {}".format(data['assigned_cust'][i-1],dist))

            model.add_link(start_point_id=0,
                        end_point_id=i,
                        distance=dist
                        )
        print(model)
        

        # Compute the links between points
        
        for i in range(1,len(data['locations'])):
            j=i+1
            for j in range(i + 1, len(data['locations'])):
                dist = self.compute_euclidean_distance(float(data['locations'][i][0]),
                                            float(data['locations'][i][1]),
                                            float(data['locations'][j][0]),
                                            float(data['locations'][j][1]),
                                            data['rc_cal_index']
                                            )
                print("Dist b/w  cust {} and {} is {}".format(data['assigned_cust'][i-1],data['assigned_cust'][j-1],dist))
                model.add_link(start_point_id= i,
                            end_point_id=j,
                            distance=dist
                            )

        
        # set parameters
        model.set_parameters(time_limit=time_resolution,
                            solver_name=solver_name_input)

        ''' If you have cplex 22.1 installed on your laptop windows you have to specify
            solver path'''
        if (solver_name_input == "CPLEX" and solver_path != ""):
            model.parameters.cplex_path = solver_path

        #model.export(instance_name)

        logging.info(f"Model is {model}")
        # solve model
        st=datetime.now()
        model.solve()
        ed=datetime.now()
        exec_time=(ed-st).total_seconds()
        # export the result
        #model.solution.export(instance_name.split(".")[0] + "_result")


        if model.solution.is_defined :
            cost = model.solution.value
            m= len(model.solution.routes)
            message= model.message
            routes = []
            for route in model.solution.routes:
                routes.append({
                    "Ids" :list(map(str, route.point_ids))    
                            })

        return cost, m, message, routes,exec_time


    def solve_files_in_directory(self,data,cust):
        
        start_time = time.time()
        cost, m, message, routes,ex_time = self.solve_demo(data,cust)
        solve_time = time.time() - start_time
        return cost,m,message,routes,ex_time





