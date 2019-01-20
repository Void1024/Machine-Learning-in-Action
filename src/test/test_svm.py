import sys
sys.path.append('..')
import algorithm.SVM as SVM
from loader.load_mnist import load_mnist
def test_svm():
    path = "../../dataset/mnist"
    k = 30
    train_x,train_y = load_mnist(path)
    test_x,test_y = load_mnist(path,'t10k')
    x = []
    y = []
    for i in range(10):
        x.extend(train_x[i * 6000:i * 6000 + 30])
        y.extend(train_y[i * 6000:i * 6000 + 30])
    SVM.train(x,y)
    check = 0
    total = int(test_y.shape[0] / 100)
    print('total %d' % total)
    for i in range(total):
        predict = SVM.classify(test_x[i])
        print("real label:%d,predict:%d" % (test_y[i],predict))
        if test_y[i] == predict:
            check = check + 1
        # if (i+1) % 10 == 0:
            # print('%d time(s),check:%d / %d' % (i + 1,check,i + 1))
    print('%f' % (check / total))

if __name__ == '__main__':
    test_svm()