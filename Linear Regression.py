# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 04:38:43 2018

@author: tunji


"""
import pandas as pd
import csv
import os
import sys
import numpy as np
from copy import deepcopy



cwd_in = os.getcwd()
fn_input = os.path.join(cwd_in, 'random.csv')
cwd_out = os.getcwd()
fn_output = os.path.join(cwd_in, 'results.csv')
sys.argv.append(fn_input)#sys.argv[1]
sys.argv.append(fn_output)#sys.argv[2]

dataset = pd.read_csv(sys.argv[1], header=None)
data_set = dataset.values

#output file
def print_error(iteration_n,weights,s_error):
    with open(sys.argv[2], 'w', newline='') as f_output:
        tsv_output = csv.writer(f_output, delimiter='\t')
        for a, b, c in zip(iteration_n,weights,s_error):
            #print_list.extend((a,b,c))            
            tsv_output.writerow([a,b,c])
    
           

#calculatef(x)
def lin_reg(row, weights):
    f = weights[0] * 1 + (weights[1] * row[0]) + (weights[2] * row[1])
    return f


def get_errors(dataset, weights):
    error_list = []
    sum_error = 0
    for i in range(len(dataset)):
        f = lin_reg(dataset[i], weights)
        error = dataset[i][2]- f
        error_list.append(error)
        sum_error += error**2
           
    return sum_error, error_list

def main(dataset, threshold,l_rate):
    weights = [0.0 for i in range(len(dataset[0]))]
    weights = np.float64(weights)
    weights_list = []
    weights_list.append(deepcopy(weights))
    iteration_n = 0
    iteration_list = []
    iteration_list.append(iteration_n)
    #threshold = 0.0001
    #l_rate = 0.0001
    s_error_list = []
    s_error_list.append(1000000000)
    s_error, error_list = get_errors(dataset, weights)
    s_error_list.append(s_error)
    
    
    for i in range(10000):
        if s_error_list[i] - s_error_list[i+1] > threshold:
            iteration_n+=1
            iteration_list.append(iteration_n)
            for row,x in zip(dataset,error_list):
                weights[0] = np.float64(weights[0] + l_rate * x)#a
                weights[1] = np.float64(weights[1] + l_rate * x * row[0])#b
                weights[2] = np.float64(weights[2] + l_rate * x * row[1])#c
            weights_list.append(deepcopy(weights))
            s_error, error_list = get_errors(dataset, weights)
            s_error_list.append(s_error)
            
        else:
            break
    #print(weights_list)
    del s_error_list[0]
    print_error(iteration_list,weights_list,s_error_list)
    return  s_error_list
    

main(data_set,0.0001,0.0001)


   