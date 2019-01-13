import sys
sys.path.append('..')

import algorithm.naive_bayes as Bayes
from loader.load_mnist import load_mnist

def test() :
    path = "../../dataset/mnist"
    dataSet,labels = load_mnist(path,'train')
    testX,testY = load_mnist(path,'t10k')
    Bayes.train(dataSet,labels)
    check = 0
    total = 0
    for i in range(len(testX)):
        predict = Bayes.classify(testX[i])
        if predict == testY[i]:
            check += 1
        total += 1
    print("total:%d time(s),hit:%d time(s),accuracy:%f." % (total,check,check/float(total)))

if __name__ == '__main__':
    test()

    
 


