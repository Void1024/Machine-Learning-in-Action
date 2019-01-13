from numpy import *
from math import log
import operator
pVect = {}
pCate = {}
# p(c|w) = p(w|c)p(c) / p(w)
# def naive_bayes_train(dataSet,category) :
#     global pVect,pCate
#     cateSet = set(category)
#     pNum = {}
#     pVect = {} # p(w|c)
#     pCate = {} # p(c)
#     for i in range(len(dataSet)):
#         if category[i] not in pVect:
#             pVect[category[i]] = zeros((2,len(dataSet[0])))
#             pCate[category[i]] = 0.0
#             pNum[category[i]] = 0
#         pVect[category[i]][0] += [0 if item == 1 else 1 for item in dataSet[i]]
#         pVect[category[i]][1] += dataSet[i]
#         pCate[category[i]] += 1 / float(len(dataSet))
#         # pNum[category[i]] += sum(dataSet[i])
#         pNum[category[i]] += 1
#     # for i in range(len(pVect)):
#     #     pVect[i] = [x/float(pNum[i])) for x in pVect[i]]
#     for label in list(pVect.keys()):
#         pVect[label][0] = [x / float(pNum[label]) for x in pVect[label][0]]
#         pVect[label][1] = [x / float(pNum[label]) for x in pVect[label][1]]
#     return pVect,pCate 

# def classify(vec) :
#     reverseVec = [1 if x == 0 else 0 for x in vec]
#     pList = {}
#     cates = list(pCate.keys())
#     for i in range(len(cates)):
#         pList[cates[i]] = sum(pVect[cates[i]][1] * vec + pVect[cates[i]][0] * reverseVec) * (pCate[cates[i]])
#         # pList[cates[i]] = sum(pVect[cates[i]] * vec) + log(pCate[cates[i]])
#         # pList[cates[i]] = log(sum(pVect[cates[i]] * vec))+log(pCate[cates[i]])
#     return sorted(pList.items(),key = operator.itemgetter(1),reverse = True)[0][0]    
    
prior_probability = {}
conditional_probability = {}
cateSet = []
def naive_bayes_train(dataSet,category) :
    global prior_probability,conditional_probability,cateSet
    cateSet = set(category)
    numCate = len(cateSet)
    numPix = len(dataSet[0])
    prior_probability = {}
    conditional_probability = {}
    for i in range(len(dataSet)):
        if category[i] not in prior_probability.keys():
            prior_probability[category[i]] = 0
        prior_probability[category[i]] += 1 / float(len(category))
        if category[i] not in conditional_probability.keys():
            conditional_probability[category[i]] = ones((len(dataSet[0]),2))
        for j in range(len(dataSet[0])):
            conditional_probability[category[i]][j][dataSet[i][j]] += 1 
    for i in cateSet:
        for j in range(len(dataSet[0])):
            conditional_probability[i][j][0] /= float(len(dataSet) + 2)
            conditional_probability[i][j][1] /= float(len(dataSet) + 2)
    return prior_probability,conditional_probability
def calculate_probability(vec,label) :
    probability = log(prior_probability[label])
    for i in range(len(vec)):
        probability += log(conditional_probability[label][i][vec[i]])
    return probability
def classify(vec) :
    max_probability = 0.0
    predict_label = 0
    for label in cateSet:
        if calculate_probability(vec,label) >= max_probability:
            predict_label = label
    return predict_label
    

    
            

        
