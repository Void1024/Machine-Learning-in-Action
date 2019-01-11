import sys
sys.path.append('..')
import algorithm.knn as KNN
from loader.load_mnist import load_mnist
def test_knn():
    path = "../../dataset/mnist"
    k = 30
    train_x,train_y = load_mnist(path)
    test_x,test_y = load_mnist(path,'t10k')
    KNN.train(train_x,train_y)
    check = 0
    total = test_y.shape[0]
    # total = 1000
    print('total %d' % total)
    for i in range(total):
        if test_y[i] == KNN.predict(test_x[i],k):
            check = check + 1
        if (i+1) % 10 == 0:
            print('%d time(s),check:%d / %d' % (i + 1,check,i + 1))
    print('%f' % (check / total))

if __name__ == '__main__':
    test_knn()