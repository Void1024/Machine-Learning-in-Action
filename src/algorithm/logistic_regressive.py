import numpy as np

weightsList = {}

# alpha = 0.000005
alpha = 0.001
maxCycle = 200

def sig_moid(inX) :
    # print('y',inX.sum()/inX.size)
    return 1.0 / (1 + np.exp(-inX))

def arg_set(inAlpha = 0.001,inCycle = 500) :
    global alpha,maxCycle
    alpha = inAlpha
    maxCycle = inCycle

def cal_weight(dataSet,labelList,goal = 1) :
    labels = np.array(labelList)
    labels[labels == goal] = -1
    labels[labels != -1] = 0
    labels[labels == -1] = 1
    global weightsList
    dataMat = np.mat(dataSet)
    labelMat = np.mat(labels).transpose()
    m,n = np.shape(dataMat)
    dataMat = np.hstack((np.ones((m,1)),dataMat))
    weightsList[goal] = np.random.rand(n + 1,1)

    for i in range(maxCycle):
        # print('x',dataMat.sum()/dataMat.size)
        
        h = sig_moid(dataMat * weightsList[goal])
        # print('z',h.sum()/h.size)
        weightsList[goal] += alpha * dataMat.transpose() * (labelMat - h)
    #     if i % 50 == 0:
            
    #         tem = h
    #         tem[tem > 0.5] = 1
    #         tem[tem <= 0.5] = 0
    #         tem -= np.mat(labels).transpose()
    #         tem[tem == 0] = 10
    #         tem[tem != 10] = 0
    #         tem[tem == 10] = 1
    #         print('i:%d,acc:%f' % (i,tem.sum() / tem.size))
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
        # print(i)
        # print(probability)
        # if probability > maxProbability:
        #     maxProbability = probability
        #     predictLabel = i
        if probability > 0.5:
            return i
    return predictLabel


def draw(dataSet,labels,) :
    pass

