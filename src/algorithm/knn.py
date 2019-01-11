from numpy import *
import operator
train_x = array([])
train_label = array([])
def normalize(x) :
    minVals = x.min(0)
    maxVals = x.max(0)
    ranges = maxVals - minVals
    normalDataset = zeros(shape(x))
    m = x.shape[0]
    normalDataset = x - tile(minVals,(m,1))
    normalDataset = normalDataset / tile(ranges,(m,1))
    return normalDataset

def train(x,label) :
    global train_x,train_label
    train_x = x
    train_label = label

def predict(x,K = 3) :
    dis = ((tile(x,(train_x.shape[0],1)) - train_x) ** 2).sum(axis = 1) ** 0.5
    dis_rank = dis.argsort()
    rank = {}
    for i in range(K):
        rank[train_label[dis_rank[i]]] = rank.get(train_label[dis_rank[i]],0) + 1
    result = sorted(rank.items(),key = operator.itemgetter(1),reverse = True)[0][0]
    return result
    