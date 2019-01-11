from math import log
import operator

def cal_shannonEnt(dataSet) :
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        label = featVec[-1]
        if label not in labelCounts:
            labelCounts[label] = 0
        labelCounts[label] += 1
    shannonEnt = 0.0
    for label in labelCounts:
        prob = float(labelCounts[label]) / numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

def split_dataSet(dataSet,axis,value) :
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def choose_best_feature2split(dataSet) :
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = cal_shannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featureList = [vec[i] for vec in dataSet]
        uniqueVals = set(featureList)
        newEntropy = 0.0
        for val in uniqueVals:
            subDataSet = split_dataSet(dataSet,i,val)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * cal_shannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majority_cnt(classList) :
    classCount = {}
    for vote in classList:
        if vote not in classCount:
            classCount[vote] = 0
        classCount[vote] += 1
    return sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)[0][0]

def create_tree(dataSet,labels) :
    classList = [item[-1] for item in dataSet]
    if len(dataSet[0]) == 1:
        return majority_cnt(classList)
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    bestFeature = choose_best_feature2split(dataSet)
    bestFeatureLabel = labels[bestFeature]
    featureValues = [item[bestFeature] for item in dataSet]
    uniqueVals = set(featureValues)
    myTree = {bestFeatureLabel : {}}
    del(labels[bestFeature])
    for val in uniqueVals:
        subLabels = labels[:]
        subDataSet = split_dataSet(dataSet,bestFeature,val)
        myTree[bestFeatureLabel][val] = create_tree(subDataSet,subLabels)
    return myTree




