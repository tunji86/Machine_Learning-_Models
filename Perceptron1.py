# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 08:07:01 2018

@author: tunji
"""
import pandas as pd
import numpy as np
import csv
#importing our data and preparing it
dataset = pd.read_csv(r'C:\Master DKE\WiSe18_19\Machine Learning\Programming Assignment 3\Example.tsv',delimiter='\t',encoding='utf-8')
del dataset['Unnamed: 3']
dataset.insert(0,column='Class',value=0)
dataset.loc[dataset['A'] == 'A', 'Class'] = 1
dataset.loc[dataset['A'] == 'B', 'Class'] = 0
del dataset['A']
data_set = dataset.values
miss_val = np.array([[1,-1.525735,1.67408]])
data_set = np.concatenate((data_set,miss_val),axis=0)
data_set[1:] = np.float64(data_set[1:])
data_set_column_size = np.ma.size(data_set,axis=1)
data_set[:,0]

#outputing our data to chosen directory
output_dir = 'C:\Master DKE\WiSe18_19\Machine Learning\Programming Assignment 3\Perceptron Solution'
def print_error(list_1, list_2):
    with open(output_dir+'/'+'example_file_errors.tsv', 'w', newline='') as f_output:
        tsv_output = csv.writer(f_output, delimiter='\t')
        tsv_output.writerow(list_1)
        tsv_output.writerow(list_2)


#definition of our perceptrons   
def perceptron(row, weights):
    activation = weights[0] + (weights[1] * row[1]) + (weights[2] * row[2])
    return 1 if activation > 0 else 0

def perceptron_2(row, weights):
    activation = weights[0] + (weights[1] * row[1]) + (weights[2] * row[2])
    return 1 if activation > 0 else 0

#iterating over our weights with updated errors
sum_error_list1=[]
sum_error_list2=[]

def train_weights(train, n_iteration):
    z=0
    weights = [0.0 for i in range(len(train[0]))]
    weights = np.float64(weights)
    weights_2 = [0.0 for t in range(len(train[0]))]
    weights_2 = np.float64(weights_2)
    
    
    for iteration in range(n_iteration):
        error_list = []
        error_list_2 = []
        z += 1
        l_rate = 1
        l_rate_2 = 1/z
        sum_error = 0
        sum_error_2 = 0 
        
        for row in train:                
            prediction = perceptron(row, weights)
            prediction_2 = perceptron_2(row,weights_2)
            error = row[0] - prediction
            error_2 = row[0] - prediction_2
            error_list.append(error)
            error_list_2.append(error_2)
            sum_error += error**2            
            sum_error_2 += error_2**2 
       
        #Class = data_set[:,0] 
        #if (Class != prediction_list).all:
        for row,x in zip(train,error_list):
            if x != 0:
                weights[0] = np.float64(weights[0] + l_rate * x)#a
                weights[1] = np.float64(weights[1] + l_rate * x * row[1])#b
                weights[2] = np.float64(weights[2] + l_rate * x * row[2])#c
            else:
                weights[0] = np.float64(weights[0])
                weights[1] = np.float64(weights[1])
                weights[2] = np.float64(weights[2])                              
        #updating annealing weight
        for row,x in zip(train,error_list_2):
            if x != 0:
                weights_2[0] = np.float64(weights_2[0] + l_rate_2 * x)#a_2
                weights_2[1] = np.float64(weights_2[1] + l_rate_2 * x * row[1])# b_2
                weights_2[2] = np.float64(weights_2[2] + l_rate_2 * x * row[2])#c
            else:
                weights_2[0] = np.float64(weights_2[0])
                weights_2[1] = np.float64(weights_2[1])
                weights_2[2] = np.float64(weights_2[2])
                
        
        print('iteration=%.0f, error=%.0f, anneal_error=%.0f' % (iteration +1, sum_error, sum_error_2))
        sum_error_list1.append(sum_error)
        sum_error_list2.append(sum_error_2)
    #print (row.dtype)        
    return weights

n_iteration = 100  
train_weights(data_set, n_iteration)
print_error(sum_error_list1,sum_error_list2)  




