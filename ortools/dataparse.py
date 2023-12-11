# Import all libraries
import math
import numpy as np
import pandas as pd
   
# Data Parse
def create_data(file_loc):
    data1=[]
    for line in open(file_loc, 'r'):
        item = line.rstrip().split()
        if len(item)!=0:
            data1.append(item)

    
    # Number of Customers
    no_cust= int(float(data1[0][0]))

    # Number of Depots
    no_depot= int(float(data1[1][0]))

    #depot coordinates
    depot_cord=[]
    ct=2
    ed_depot=ct+no_depot
    for i in range(ct,ed_depot):
        depot_cord.append(tuple(data1[i]))

    #Customer coordinates
    cust_cord=[]
    ed_cust= ed_depot+no_cust
    for j in range(ed_depot,ed_cust):
        cust_cord.append(tuple(data1[j]))

    # Vehicle Capacity
    vehicle_cap= int(data1[ed_cust][0])
    vehicle_cap=[vehicle_cap]*no_depot

    #Depot capacities #B1
    depot_cap=[]
    start=ed_cust+1
    end=start+no_depot
    for k in range(start,end):
        depot_cap.append(int(float(data1[k][0])))
   

    # Customer Capacities
    dem_end=end+no_cust
    cust_dem=[]
    for l in range(end,dem_end):
        cust_dem.append(int(float(data1[l][0])))

    # # Opening Cost of Depots #B1
    open_dep_cost=[]
    cost_end=dem_end+no_depot
    for x in range (dem_end,cost_end):
        open_dep_cost.append(int(float(data1[x][0])))
    


    route_cost=int(float(data1[cost_end][0]))

    rc_cal_index=int(data1[cost_end+1][0])


    return [no_cust,no_depot,depot_cord,cust_cord,vehicle_cap,depot_cap,cust_dem,open_dep_cost,route_cost,rc_cal_index]



def dist_calc(cord_set1,cord_set2,rc_cal_index):
    
    distances = {}
    # make this one depot coordiantes
    for from_counter, from_node in enumerate(cord_set1):
        distances[from_counter] = {}
        # customer coordinates
        for to_counter, to_node in enumerate(cord_set2):
                # Euclidean distance
                if rc_cal_index==0:
                    distances[from_counter][to_counter] = (int(100*(math.hypot(float(from_node[0]) - float(to_node[0]),float(from_node[1]) - float(to_node[1])))))
                else:
                     distances[from_counter][to_counter] = (int(math.hypot(float(from_node[0]) - float(to_node[0]),float(from_node[1]) - float(to_node[1]))))
    return distances

def new_dist_calc(cord_set,rc_cal_index):
    distances = {}
    
    sum1=0
    for j in (cord_set):
        distances[j]={}
        i=0
        for i in (cord_set[j]):
                if rc_cal_index==0:
                    distances[j][i] = (100*(math.sqrt((cord_set[j][i][0])**2)+(cord_set[j][i][1])**2))
                    sum1=sum1+distances[j][i]
                else:
                    distances[j][i] = (math.sqrt((cord_set[j][i][0])**2)+(cord_set[j][i][1])**2)
                    sum1=sum1+distances[j][i]
    
    return distances,sum1

def normalize_coord(cord_set1,cord_set2,rc_cal_index):
    coor=pd.DataFrame()
    mod_coord = {}
    mod_coord1={}
    sum_dist={}
    dist_fac_cust={}
    # depot coordiantes
    fi_list=[]
    for j in range(len(cord_set1)):
        mod_coord[j]={}
        max_x=0
        min_x=0
        max_y=0
        min_y=0
        # customer coordinates
        for i in range(len(cord_set2)):
                mod_coord[j][i] = (float(cord_set2[i][0]) - float(cord_set1[j][0])),(float(cord_set2[i][1]) - float(cord_set1[j][1]))
                x = (float(cord_set2[i][0]) - float(cord_set1[j][0]))
                max_x=max(max_x,x)
                min_x=min(min_x,x)
                y = (float(cord_set2[i][1]) - float(cord_set1[j][1]))
                max_y=max(max_y,y)
                min_y=min(min_y,y)
        fi=max((max_x-min_x),(max_y-min_y))
        fi_list.append(fi)
        print(max_x,min_x,max_y,min_y)

    #_ = [mod_coord.__setitem__(j, {i: [mod_coord[j][i]/fi_list[j]] for i in mod_coord[j]}) for j in mod_coord]
    for j in mod_coord:
         for i in mod_coord[j]:
              mod_coord[j][i]=[mod_coord[j][i][0] / fi_list[j] ,mod_coord[j][i][1] / fi_list[j]]
              
    dist_fac_cust,big_m=new_dist_calc(mod_coord,rc_cal_index)
    coor['normal']=mod_coord.values()
    #coor['normal']=mod_coord1.values()
    print("Big M value:",big_m)
    print("Normalization factors for all depots",fi_list)
    return coor,dist_fac_cust,big_m,fi_list

def norm_data(cord_set1,cord_set2,veh_cap,cust_dem,rc_cal_index):
    facility_dict={}
    ans=normalize_coord(cord_set1,cord_set2,rc_cal_index)
    mod_coord1=ans[0]['normal']
    big_m=ans[2]
    cost_norm_factor=ans[3]
    for j in range(len(cord_set1)):
        norm_df=pd.DataFrame()
        norm_cust_dem=[cust_dem[i]/veh_cap[j] for i in range(len(cust_dem))]
        
        print(mod_coord1[j],'\n')
        facility_dict[j]=pd.DataFrame()
        mod_coord_x=([mod_coord1[j][i][0] for i in range(len(cord_set2))] )
        mod_coord_y=([mod_coord1[j][i][1] for i in range(len(cord_set2))] )
        norm_df['x']=mod_coord_x
        norm_df['y']=mod_coord_y
        norm_df['dem']=norm_cust_dem
        facility_dict[j]=norm_df
    return facility_dict, big_m,cost_norm_factor
    
    
    