
"""
Created on Thu Dec 20 20:08:23 2018

@author: tunji
"""
import pandas as pd
import numpy as np
import random
import csv
import math


#importing our data and preparing it
dataset = pd.read_csv(r'C:\Master DKE\WiSe18_19\Machine Learning\Programming Assignment 4\Example.tsv',delimiter='\t',encoding='utf-8')
del dataset['Unnamed: 3']
dataset.insert(0,column='Class',value=0)
dataset.loc[dataset['A'] == 'A', 'Class'] = 1
dataset.loc[dataset['A'] == 'B', 'Class'] = 0
del dataset['A']
data_set = dataset.values
miss_val = np.array([[1,-1.525735,1.67408]])
data_set = np.concatenate((miss_val,data_set),axis=0)
data_set[1:] = np.float64(data_set[1:])
data_set_column_size = np.ma.size(data_set,axis=1)

#outputing to tsv file
output_dir = 'C:\Master DKE\WiSe18_19\Machine Learning\Programming Assignment 4'
def print_error(list_1, list_2, prior_1, prior_2, num_of_misclassed):
    misclassed_total = [num_of_misclassed]
    with open(output_dir+'/'+'NB_Example_Solution.tsv', 'w', newline='') as f_output:
        tsv_output = csv.writer(f_output, delimiter='\t')
        tsv_output.writerow(list_2 + prior_2)
        tsv_output.writerow(list_1 + prior_1)
        tsv_output.writerow(misclassed_total)

#split data into train and test sets
def splitDataset(data, ratio):
    trainSize = int(len(data) * ratio)
    train_set = []
    test_set = list(data)
    while len(train_set) < trainSize:
        index = random.randrange(len(test_set))
        train_set.append(test_set.pop(index))
    return [train_set, test_set]


def class_separation(data):
    class_separated = {}
    for i in range(len(data)):
        data_point = data[i]
        if (data_point[0] not in class_separated):
            class_separated[data_point[0]] = []
        class_separated[data_point[0]].append(data_point)
    return class_separated


def mean(column_data):
    return sum(column_data)/float(len(column_data))

def variance_(column_data):
    avg = mean(column_data)
    variance = sum([(x-avg)**2 for x in column_data])/float(len(column_data)-1)
    return variance

def class_prior(dataset):
    prior_list = []
    for i in set(dataset[:, 0]):
        x = dataset[:, 0] == i
        count_ = np.count_nonzero(x == True)
        prior = float(count_/len(dataset))
        prior_list.append(prior)
    return prior_list

def calc_mean_std(dataset):
    mean_std = [(mean(column_data), variance_(column_data)) for column_data in zip(*dataset)]
    del mean_std[0]
    return mean_std


def classify(dataset, training_set, test_set):
    #get entire dataset mean and STD
    class_separated = class_separation(dataset)
    mean_std = {}
    for classValue, instances in class_separated.items():
        mean_std[classValue] = calc_mean_std(instances)
    list_1 = mean_std.get(0.0)
    list_2 = mean_std.get(1.0)
    
    #get training set mean and STD for predictions********************************
    train_class_separated = class_separation(training_set)
    train_mean_std = {}
    for classValue, instances in train_class_separated.items():
        train_mean_std[classValue] = calc_mean_std(instances)
    
    
    #prior of classes*************************************************************
    prior_list = class_prior(dataset)
    prior_1 = [prior_list[0]]
    prior_2 = [prior_list[1]]
    
    #do predictions***************************************************************
    predictions = []
    for i in range(len(test_set)):
        result = predict(dataset, train_mean_std, test_set[i])
        predictions.append(result)
    #count misclassified
    num_of_misclassed = 0
    for i in range(len(test_set)):
        if test_set[i][0] != predictions[i]:
            num_of_misclassed += 1
    print_error(list_1, list_2,prior_1,prior_2, num_of_misclassed)
    return num_of_misclassed

#likelyhood for each attribute value per class
def calculate_probability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

#calculating class probabilities
def probability_by_class(dataset, mean_std, test_data):
    #get class priors
    prior_list = class_prior(dataset)
    #calculate probability
    probability = {}
    for (class_, class_mean_std),prior_value in zip(mean_std.items(),prior_list):
        probability[class_] = 1
        for i in range(len(class_mean_std)):
            mean, stdev = class_mean_std[i]
            probability[class_] *= calculate_probability(test_data[i], mean, stdev)
        probability[class_] = probability[class_] * prior_value
    return probability


#predict function
def predict(dataset, mean_std, inputVector):
    probabilities = probability_by_class(dataset, mean_std, inputVector)
    chosen_class, max_prob = None, -1
    for class_, probability in probabilities.items():
        if chosen_class is None or probability > max_prob:
            max_prob = probability
            chosen_class = class_
    return chosen_class


training_set, test_set = splitDataset(data_set, 0.75)
classify(data_set, training_set, test_set)






