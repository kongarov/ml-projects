import _pickle as cp
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


class NBC:

#initialize the model
    def __init__(self,feature_types,  pseudoCount=1.0 ,num_classes=None, class_labels=None):
        if num_classes is None:
            num_classes=len(class_labels)

        if class_labels is None:
            class_labels=np.arange(num_classes)

        def feat_trans(x):
            if (x=='b'):
                return [0,1]
            return x

        self.num_classes=num_classes
        self.class_labels=class_labels
        self.features_types=list(map(feat_trans,feature_types))
        self.D=len(feature_types)
        self.pseudoCount=pseudoCount

#given a list of values returns a log of pdf of Normal Dist
    def find_Gaussian(xs):
        n=np.mean(xs)
        v=np.var(xs)
        if v == 0:
            v = 1e-6
        return NBC.get_logGaussian(n,v)


#given a list of values returns log of pmf of MN Dist
    def find_MN(pseudoCount ,posV, values):
        den: int =len(values)+len(posV)*pseudoCount
        dist = Counter(values)
        ans= dict()
        for i in posV:
            ans.update({i:np.log((dist[i]+pseudoCount)/den)})
        return NBC.get_logMN(ans)

#given paramat for Normal Dist returns log of pdf
    def get_logGaussian(mn,var):
        con = -0.5*np.log(2*np.pi*var)
        def f(a):
            return con-((a-mn)**2)/var
        return f

#given paramt for MN Dist returns log of pmf
    def get_logMN(dic):
        def f(a):
            return dic[a]
        return f

#splits X by corresponding values of y
    def split_by_label(self, X,y):
        N=len(X)
        assert(N==len(y))
        ans=dict()
        for i in range(N):
            if ans.get(y[i]) is None:
                ans[y[i]] = [X[i]]
            else:
                ans[y[i]].append(X[i])
        for yi in self.class_labels:
            if ans.get(yi) is None:
                ans[yi]=[]
        return ans


#fit a model for subset of the input
    def fit_model_feature_label(self,X):
        A = np.transpose(X)
        ans = []
        for i in range(self.D):
            if 'r' == self.features_types[i]:
                if len((np.asarray(X)).shape)!= 2:
                    cur_pdf=NBC.get_logGaussian(0,1)
                else:
                    cur_pdf=NBC.find_Gaussian(A[i])
            else :
                 cur_pdf=NBC.find_MN(self.pseudoCount,self.features_types[i],A[i])
            ans.append(cur_pdf)
        return ans

#find the actual model
#represented by function dist_y log pmf of y
#and dist_x dict with keys values of y and values list (length D) functions for each param in the input
    def fit(self,X,y):
        self.dist_y=NBC.find_MN(self.pseudoCount, self.class_labels, y)
        splited = self.split_by_label(X,y)
        ans = dict()
        self.dist_X = {yi:self.fit_model_feature_label(splited[yi])for yi in self.class_labels}


#find log of prob for one new data point
    def prob1(self,xs,y):
         ans= self.dist_y(y)
         for i in range(self.D):
             ans += self.dist_X[y][i](xs[i])
         return ans

#Predict for single input
    def predict1(self,xs):
        maxprob= - np.infty
        ans=-1
        for i in self.class_labels:
            prob = self.prob1(xs, i)
            if maxprob < prob:
                ans = i
                maxprob=prob
        return ans

#Predict multiple inputs
    def predict(self,X_new):
        return np.asarray(list(map (self.predict1, X_new)))



X = np.array([[0.5, 0, 2],[0.7, 0, 1],[0, 1, 0],[-2, 0, 1],[3, 1, 2],[1, 1, 1],[-1.5, 0, 0]])
y = ['a', 'b', 'b', 'c', 'b', 'a', 'b']
o = NBC (feature_types = ['r','b', [0, 1, 2]], class_labels = ['a', 'b', 'c'])
o.fit(X, y)
print(o.predict(X))
print(np.mean(y==o.predict(X)))


class LR:
    def __init__(self,lamb=1.0):
        self.model = LogisticRegression(solver='liblinear',multi_class = 'ovr',C=1/lamb)
    def fit(self, X, y):
         self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict(X)

# C is the inverse regularisation strength => C = 1/λ

X = np.array([[0.5, 0, 2], [0.7, 0, 1],[0, 1, 0],[-2, 0, 1],[3, 1, 2], [1, 1, 1], [-1.5, 0, 0]])
y = ['a', 'b', 'b', 'c', 'b', 'a', 'b']
o = LR (0.1)
o.fit(X, y)
print(np.mean(y==o.predict(X)))

from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris['data'], iris['target']

def exper(model,X,y,times):
    N, D = X.shape
    Ntrain=int(N*0.8)
    cum_err=np.zeros(10)
    i=0
    while i<times:
        shuffler = np.random.permutation(N)
        Xtrain = X[shuffler[:Ntrain]]
        ytrain = y[shuffler[:Ntrain]]
        Xtest = X[shuffler[Ntrain:]]
        ytest = y[shuffler[Ntrain:]]
        for j in range(10):
            train_s = int(0.1*(j+1)*Ntrain)
            Xtrain_cur = Xtrain[:train_s]
            ytrain_cur = ytrain[:train_s]
            model.fit(Xtrain_cur, ytrain_cur)
            cum_err[j] += np.mean(ytest == model.predict(Xtest))
        i += 1
    return list(map(lambda x:(1-x/times), cum_err))

def plot(nbc,lr,X,y,tname,times=400):
    err_nbc=exper(nbc,X,y,times)
    err_lr=exper(lr,X,y,times)
    x_axis=list(range(10,101,10))
    plt.plot(x_axis,err_nbc, label = "NBC error")
    plt.plot(x_axis,err_lr, label = "LR error")
    plt.xlabel('% training data used')
    plt.ylabel('Test error')
    plt.title(tname)
    plt.legend()
    plt.show()

from sklearn.datasets import load_iris
iris = load_iris()
X_iris, y_iris = iris['data'], iris['target']
nbc_iris = NBC(feature_types = ['r', 'r', 'r', 'r'], class_labels = [0, 1, 2])
lr_iris = LR(0.1)
plot(nbc_iris,lr_iris, X_iris, y_iris, 'iris',500)

import pickle as cp
X_vot, y_vot = cp.load(open('voting.pickle', 'rb'))
nbc_vot = NBC(feature_types = np.full(16, 'b'), class_labels = [0, 1],pseudoCount = 1e-3)
lr_vot = LR(0.1)
plot(nbc_vot, lr_vot, X_vot, y_vot,'voting')

def fill(X):
    for i in range(X.shape[1]):
        col = list(X[:, i])
        mn = np.mean(np.array([x for x in col if x != 2]))
        X[:, i] = np.asarray(list(map((lambda x: mn if x == 2 else x),col)))
    return X

X_votf, y_votf = cp.load(open('voting-full.pickle', 'rb'))
X_votf=fill(X_votf)
nbc_votf = NBC(feature_types = np.full(16, 'r'), class_labels = [0, 1],pseudoCount = 1e-3)
lr_votf = LR(0.1)
plot(nbc_votf, lr_votf, X_votf, y_votf,'voting-full')


