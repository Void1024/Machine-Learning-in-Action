import numpy as np

C = 200
toler = 0.0001
maxIter = 10000
kTup = ('rbf',10)

classifier = {}

class optStruct:
    def __init__(self,dataMatIn,classLabels,C,toler,kTup):
        self.x = dataMatIn
        self.y = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m,1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m,2)))
        self.K = np.mat(np.zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.x,self.x[i,:],kTup)
def kernelTrans(X,A,kTup):
    m,n = np.shape(X)
    K = np.mat(np.zeros((m,1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K / (-1 * kTup[1]**2))
    else:
        raise NameError('Kernel is not recognized')
    return K

def selectJrand(i,m):
    j = i
    while j == i:
        j = int(np.random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj < L:
        aj = L
    elif aj > H:
        aj = H
    return aj

def calcEk(oS,k):
    fXk = float(np.multiply(oS.alphas,oS.y).T * oS.K[:,k]) + oS.b
    Ek = fXk - float(oS.y[k])
    return Ek
def updateEk(oS,k):
    oS.eCache[k] = [1,calcEk(oS,k)]
def selectJ(i,oS,Ei):
    maxK = -1
    maxDelta = 0.0
    Ej = 0
    oS.eCache[i] = [1,Ei]
    validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]
    if len(validEcacheList) > 1:
        for k in validEcacheList:
            if i == k:
                continue
            Ek = calcEk(oS,k)
            delta = abs(Ei - Ek)
            if delta > maxDelta:
                maxDelta = delta
                maxK = k
                Ej = Ek
    else:
        maxK = selectJrand(i,oS.m)
        Ej = calcEk(oS,maxK)
    return maxK,Ej
def innerL(i,oS):
    Ei = calcEk(oS,i)
    if ((oS.y[i] * Ei < -oS.tol) and (oS.alphas[i] < C) or \
        (oS.y[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i,oS,Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if oS.y[i] == oS.y[j]:
            H = min(oS.C,oS.alphas[j] + oS.alphas[j])
            L = max(0,oS.alphas[i] + oS.alphas[j] - oS.C)
        else:
            H = min(oS.C,oS.C + oS.alphas[j] - oS.alphas[i])
            L = max(0,oS.alphas[j] - oS.alphas[i])
        if L == H:
            # print("L == H")
            return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]
        if eta >= 0:
            # print("eta >= 0")
            return 0
        oS.alphas[j] -= oS.y[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS,j)
        if(abs(oS.alphas[j] - alphaJold) < 0.00001):
            # print("j not moving enough")
            return 0
        oS.alphas[i] += (oS.y[j] / oS.y[i]) * (alphaJold - oS.alphas[j])
        updateEk(oS,i)
        b1 = oS.b - Ei - oS.y[i] * (oS.alphas[i] - alphaIold) * oS.K[i,i] -\
            oS.y[j] * (oS.alphas[j] - alphaJold) * oS.K[i,j]
        b2 = oS.b - Ej - oS.y[i] * (oS.alphas[i] - alphaIold) * oS.K[i,j] -\
            oS.y[j] * (oS.alphas[j] - alphaJold) * oS.K[j,j]
        if (oS.alphas[i] > 0) and (oS.alphas[i] < C):
            oS.b = b1
        elif (oS.alphas[j] > 0) and (oS.alphas[j] < C):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

def smoP(dataMatIn,classLabels,C,toler,maxIter,kTup = ('lin',0)):
    oS = optStruct(np.mat(dataMatIn),np.mat(classLabels).transpose(),C,toler,kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while ((iter < maxIter) and (alphaPairsChanged > 0) or (entireSet)):
        print(iter)
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)
            iter += 1
        else:
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
            iter += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
    return oS.b,oS.alphas

# def calcWs(alphas,dataArr,classLabels):
#     X = np.mat(dataArr)
#     labelMat = np.mat(classLabels).transpose()
#     m,n = np.shape(x)
#     w = np.zeros((n,1))
#     for i in range(m):
#         w += np.multiply(alphas[i] * labelMat[i],X[i,:].T)
#     return w


def train(dataSet,labels):
    global classifier
    labelSet = list(set(labels))
    for label in labelSet:
        bin_labels = class_binarize(labels,label)
        alphas,b = train_classifier(dataSet,bin_labels)

        svIndex = np.nonzero(alphas.A > 0)[0]
        supportV = np.array(dataSet)[svIndex]
        svLabel = np.array(bin_labels)[svIndex]
        svAlpha = alphas[svIndex]
        classifier[label] = (alphas,b,supportV,svLabel,svAlpha)
    return classifier

def class_binarize(labels,label):
    new_label = [1 if l == label else -1 for l in labels]  
    return new_label

def train_classifier(dataSet,labels):
    b,alphas = smoP(dataSet,labels,C,toler,maxIter,kTup)
    return alphas,b


def predict(svm,dataVec):
    return kernelTrans(svm[2],np.mat(dataVec),kTup).T * np.multiply(svm[3],svm[4].T).T + svm[1]
def classify(dataVec):
    l = 0
    s = predict(classifier[0],dataVec)
    for i in range(1,10):
        svm = classifier[i]
        r = predict(svm,dataVec)
        if r > s:
            s = r
            l = i
    return l

