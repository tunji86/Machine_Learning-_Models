# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 15:34:05 2019

@author: tunji
"""
import pandas as pd
import math
from operator import itemgetter
import csv
import os
import sys


cwd_in = os.getcwd()
fn_input = os.path.join(cwd_in, 'Example.tsv')
cwd_out = os.getcwd()
fn_Prog = os.path.join(cwd_in, 'Prog.tsv')
fn_Proto = os.path.join(cwd_in, 'Proto.tsv')


sys.argv.append(fn_input)#sys.argv[1]
sys.argv.append(fn_Prog)#sys.argv[2]
sys.argv.append(fn_Proto)#sys.argv[3]


#get dataset

#cwd = sys.argv[1]

dataset = pd.read_csv(sys.argv[1], header=None, delim_whitespace=True)
del dataset[0]
data_set = dataset.values
#dataset = pd.read_table(r'C:\Master DKE\WiSe18_19\Machine Learning\Programming Assignment 6\Program Files\Example.tsv',delim_whitespace=True,header=None)

#print out output tsv
def print_error(Prog_list_1,Proto_list_2):
    #a = misclassed.
    with open(sys.argv[2], 'w', newline='') as f_output:
        tsv_output = csv.writer(f_output, delimiter='\t')
        tsv_output.writerow(Prog_list_1)
        
    with open(sys.argv[3], 'w', newline='') as f_output:
        tsv_output = csv.writer(f_output, delimiter='\t')
        tsv_output.writerow(Proto_list_2)

#distance function
def euclideanDistance(instance1, instance2):
    distance = 0
    for x in range(len(instance1)):
        distance += ((instance1[x] - instance2[x]))**2
    return math.sqrt(distance)

    
#optimizer calculator
def distance_error(cluster_point, centroid):
    distance = 0
    for x in range(len(cluster_point)):
        distance += ((cluster_point[x] - centroid[x]))**2
    return distance
    
def get_centroid(cluster):
    x = []
    y = []
    for i in cluster:
        x.append(i[0])
        y.append(i[1])
    a = sum(x)/len(cluster)
    b = sum(y)/len(cluster)
    centroid = (a,b)
    
    return centroid

            
def cluster_sse(cluster):
    prototypes = [(0,5),(0,4),(0,3)]
    centroid = get_centroid(cluster)
    prototypes.append(centroid)#to get all centroids for print
    sse = 0
    for i in range(len(cluster)):
        sse += distance_error(cluster[i], centroid)
    return sse
    
    
    
    

def get_clusters(dataset,current_prototype):
    #current_prototype = [(0,5),(0,4),(0,3)]
    cluster_1 = []
    cluster_2 = []
    cluster_3 = []
    
    for i in range(len(dataset)):
        dist_list = []
        dis1 = euclideanDistance(dataset[i],current_prototype[0])
        dist_list.append(dis1)
        dis2 = euclideanDistance(dataset[i],current_prototype[1])
        dist_list.append(dis2)
        dis3 = euclideanDistance(dataset[i],current_prototype[2])
        dist_list.append(dis3)
        min_dis_index = min(enumerate(dist_list), key=itemgetter(1))[0] 
        
        #print(dist_list)
        if min_dis_index == 0:
            cluster_1.append(dataset[i])
        elif min_dis_index == 1:
            cluster_2.append(dataset[i])
        else:
            cluster_3.append(dataset[i])
    #get new centroids
    centroid_1 = get_centroid(cluster_1)
    centroid_2 = get_centroid(cluster_2)
    centroid_3 = get_centroid(cluster_3)
    new_prototypes = []
    new_prototypes = [centroid_1,centroid_2,centroid_3]
    #get cluster sse
    sse_1 = cluster_sse(cluster_1)
    sse_2 = cluster_sse(cluster_2)
    sse_3 = cluster_sse(cluster_3)
    #print(sse_1)
    
    total_sse = sse_1 + sse_2 + sse_3
    
    
    return total_sse, new_prototypes

def main_(dataset):
    sse_list = []
    sse_list.append(0)
    prototypes = [(0,5),(0,4),(0,3)]
    sse, centroids = get_clusters(dataset,prototypes)
    sse_list.append(sse)
    prototypes.append(centroids)
    #prev_sse = sse
    
    for i in range(1000):
        if sse_list[i] != sse_list[i+1]:
            sse, centroids = get_clusters(dataset,centroids)
            sse_list.append(sse)
            prototypes.append(centroids)
        else:
            break
    del sse_list[0]
    
    print_error(sse_list,prototypes)    
    return sse_list, prototypes
            
main_(data_set)   

