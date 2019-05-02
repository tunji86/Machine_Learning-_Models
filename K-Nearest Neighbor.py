# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 12:54:41 2018

@author: tunji
"""

import pandas as pd
import numpy as np
import csv
import math
import operator 

#importing our data and preparing it
dataset = pd.read_table(r'C:\Master DKE\WiSe18_19\Machine Learning\Programming Assignment 5\ExampleShuffled.tsv',delim_whitespace=True,header=None)
data_set = dataset.values

#outputing CSV
output_dir = 'C:\Master DKE\WiSe18_19\Machine Learning\Programming Assignment 5'
def print_error(misclassed,CB):
    #a = misclassed.
    with open(output_dir+'/'+'KNN_Example1_Solution.tsv', 'w', newline='') as f_output:
        tsv_output = csv.writer(f_output, delimiter='\t')
        tsv_output.writerow(misclassed)
        tsv_output.writerow(CB)
        
#distance formular
def euclideanDistance(instance1, instance2):
    distance = 0
    for x in range(len(instance1)-1):
        distance += pow((instance1[x+1] - instance2[x+1]), 2)
    return math.sqrt(distance)  
        
#calculate nearest neighbors
def getNeighbors(CB, datapoint, k):
    data_dist = []
    #length = len(testInstance)-1
    for x in range(len(CB)):
        dist = euclideanDistance(datapoint, CB[x])
        data_dist.append((CB[x],dist))
    data_dist.sort(key=operator.itemgetter(1))
    
    neighbors = []
    for i in range(k):
        #print(data_dist[i])
        neighbors.append(data_dist[i][0])
    return neighbors



#calculate weight and return prediction for i
def classify_i(neighbors, datapoint):
    #sorting neighbors again just to be sure
    neighbor_dist = []
    for x in range(len(neighbors)):
        dist = euclideanDistance(datapoint, neighbors[x])
        neighbor_dist.append((neighbors[x],dist))
    neighbor_dist.sort(key=operator.itemgetter(1))
    sorted_neighbors = []
    for i in range(len(neighbor_dist)):
        sorted_neighbors.append(neighbor_dist[i][0])
    #**************************************************************************    
    k = len(sorted_neighbors)-1
    A_weights = []
    B_weights = []
    #prediction = None
    dk = euclideanDistance(datapoint,sorted_neighbors[k])
    d1 = euclideanDistance(datapoint,sorted_neighbors[0])
    for i in range(len(sorted_neighbors)):
        di = euclideanDistance(datapoint,sorted_neighbors[i])
        numerator = dk - di
        denominator = dk - d1
        if dk != d1:
            weight_i = numerator/denominator
        else:
            weight_i = 1
        
        #include weight in weight list A or B
        if sorted_neighbors[i][0] == 'A':
            A_weights.append(weight_i)
        else:
            B_weights.append(weight_i)
    #print(A_weights)
    #print(B_weights)
    #determine prediction
    if sum(A_weights) > sum(B_weights):
        prediction = 'A'
    else:
        prediction = 'B'
    return prediction



#call for all predictions
def main(dataset):
    misclassed_array = []
    CB_four_NN = []
    for k in (2,4,6,8,10):
        CB = []
        CB.append(dataset[0])
        data_sliced = np.delete(dataset,slice(0,1),axis=0)
        remaining_points = []
        #create Case Base
        for i in range(len(data_sliced)):
            if len(CB) == 1:
                if CB[0][0] != data_sliced[i][0]:
                    CB.append(data_sliced[i])
                else:
                    remaining_points.append(data_sliced[i])
                    
            elif len(CB) > 1 and len(CB) < k:
                #if k == 4:
                    #print('****')
                j = len(CB)
                neighbors = getNeighbors(CB, data_sliced[i], j)
                prediction = classify_i(neighbors,data_sliced[i])
                #print(prediction, data_sliced[i][0])
                if prediction == data_sliced[i][0]:
                #if neighbors[0][0] == data_sliced[i][0]:
                    #print('In remain')
                    remaining_points.append(data_sliced[i])
                else:
                    #print('In CB')
                    CB.append(data_sliced[i])
                                    
            else:
                neighbors = getNeighbors(CB, data_sliced[i], k)
                prediction = classify_i(neighbors,data_sliced[i])
                if prediction == data_sliced[i][0]:
                    remaining_points.append(data_sliced[i])
                else:
                    CB.append(data_sliced[i])        
        
        #classify remaining points with different k-values
        num_of_misclassed = 0
        for i in range(len(remaining_points)):
            #f = len(CB)
            neighbors = getNeighbors(CB, remaining_points[i], k)
            prediction = classify_i(neighbors,remaining_points[i])
            if prediction != remaining_points[i][0]:
            #if neighbors[0][0] != remaining_points[i][0]:
                num_of_misclassed+=1
        #Add number of misclassed for k to misclassed array
        misclassed_array.append(num_of_misclassed)
        if k == 4:
            CB_four_NN = CB
            
    print(len(CB_four_NN))        
    print_error(misclassed_array,CB_four_NN) 
    return CB_four_NN, misclassed_array

main(data_set) 

