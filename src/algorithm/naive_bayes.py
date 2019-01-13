from numpy import *
from math import log

condiction_probability = {}
prior_probability = {}
labelSets = []

def train(dataSet,labels) :
    global condiction_probability,prior_probability,labelSets
    labelSets = list(set(labels))
    
    for i in range(len(dataSet)):
        label = labels[i]
        vec = dataSet[i]
        if label not in prior_probability.keys():
            prior_probability[label] = 0
            condiction_probability[label] = zeros((len(vec),2))
        prior_probability[label] += 1 / len(dataSet)
        for j in range(len(vec)):
            condiction_probability[label][j][vec[j]] += 1
    for label in labelSets:
        for i in range(len(dataSet[0])):
            p0 = condiction_probability[label][i][0]
            p1 = condiction_probability[label][i][1]
            condiction_probability[label][i][0] = float(p0 + 1) / float(p0 + p1 + 2)
            condiction_probability[label][i][1] = float(p1 + 1) / float(p0 + p1 + 2)
    return prior_probability,condiction_probability

def calculate_probability(vec,label) :
    probability = log(prior_probability[label])
    for i in range(len(vec)):
        probability += log(condiction_probability[label][i][vec[i]])
    return probability

def classify(vec) :
    max_probability = calculate_probability(vec,labelSets[0])
    predict_label = labelSets[0]
    for label in labelSets:
        probability = calculate_probability(vec,label)
        if probability >= max_probability:
            max_probability = probability
            predict_label = label
    return predict_label



    
            

        
