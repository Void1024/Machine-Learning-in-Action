import sys
sys.path.append('..')

import algorithm.decision_tree as Tree

def create_dataSet() :
    dataSet = [
        [1,1,'yes'],
        [1,1,'yes'],
        [1,0,'no'],
        [0,1,'no'],
        [0,1,'no']
    ]
    labels = ['no surfacing','flippers']
    return dataSet,labels

def create_testVec() :
    vec = [0,2]
    return vec
def test() :
    d,l = create_dataSet()
    theTree = Tree.create_tree(d,l)
    testVec = create_testVec()
    classLabel = Tree.classify(theTree,testVec,l)
    print('testVec:')
    print(testVec)
    print('classify result:%s' % classLabel)

if __name__ == '__main__':
    test()