import numpy as np

weightsList = {}

alpha = 0.01
maxCycle = 300

def sig_moid(inX) :
    return 1.0 / (1 + np.exp(-inX))

def arg_set(inAlpha = 0.001,inCycle = 500) :
    global alpha,maxCycle
    alpha = inAlpha
    maxCycle = inCycle

def cal_weight(dataSet,labelList,goal = 1) :
    labels = np.array(labelList)
    labels[labels == goal] = 10
    labels[labels != 10] = 0
    labels[labels == 10] = 1
    global weightsList
    dataMat = np.mat(dataSet)
    labelMat = np.mat(labels).transpose()
    m,n = np.shape(dataMat)
    dataMat = np.hstack((np.ones((m,1)),dataMat))
    weightsList[goal] = np.random.rand(n + 1,1)

    for i in range(maxCycle):
        h = sig_moid(dataMat * weightsList[goal])
        weightsList[goal] += alpha * dataMat.transpose() * (labelMat - h)

        if i % 50 == 0:    
            tem = dataMat * weightsList[goal]
            tem[tem > 0.5] = 1
            tem[tem <= 0.5] = 0
            tem -= np.mat(labels).transpose()
            tem[tem == 0] = 10
            tem[tem != 10] = 0
            tem[tem == 10] = 1
            print('i:%d,acc:%f' % (i,tem.sum() / tem.size))
    print(goal)
    return weightsList[goal]

def train(dataSet,labels) :
    labellist = list(set(labels))
    for label in labellist:
        cal_weight(dataSet,labels,label)

def classify(testVec) :
    maxProbability = -1
    predictLabel = -1
    for i in range(10):
        probability = sig_moid(np.mat(testVec) * weightsList[i])[0][0]
        if probability > 0.5:
            return i
    return predictLabel


