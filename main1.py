import math
import _pickle as cp
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
X, y = cp.load(open('winequality-white.pickle', 'rb'))

#Split the data into training and testing parts
N, D = X.shape
N_train = int(0.8 * N)
N_test = N - N_train
X_train = X[:N_train]
y_train = y[:N_train]
X_test = X[N_train:]
y_test = y[N_train:]


#Draw a histogram
y_byGr= Counter(y)
plt.bar(y_byGr.keys(), y_byGr.values())
plt.xlabel('Wine quality')
plt.ylabel('Number of train tests')
plt.show()



#Trivial Estimate
est1=np.mean(y_train)
print("The estimation is", est1)
err1=np.mean((y_train-est1)**2)
print("The mean sq train error is", err1)
err1=np.mean((y_test-est1)**2)
print("The mean sq test error is", err1)



#Normalizing parametres
means_X= np.mean(X_train,axis=0)
devs_X =np.sqrt(np.var(X_train, axis=0))


def normalise(X,a,d):
    return (X - a)/d


Y= normalise(X, means_X, devs_X)
Y=np.insert(Y,0,1,axis=1)


def lin_mod(X,y):
    A=np.transpose(X)
    return (np.linalg.inv(A.dot(X))).dot(A.dot(y))

#Find the training and test error using LLE
def find_errors(N_train):
    Y_train = Y[:N_train]
    y_train = y[:N_train]
    Y_test = Y[N_train:]
    y_test = y[N_train:]
    w=lin_mod(Y_train,y_train)
    est2_tr=Y_train.dot(w)
    est2_test=Y_test.dot(w)
    tr_err=np.mean((y_train-est2_tr)**2)
    test_err=np.mean((y_test-est2_test)**2)
    return (tr_err,test_err)

(tr_err,test_err)=find_errors(N_train)

print("The mean sq train error is", tr_err)
print("The mean sq test error is", test_err)


x_plot=list(range (20, 600, 20))

#Find the errors
train_errs=[]
test_errs=[]
for i in x_plot:
    (tr_err,test_err)=find_errors(i)
    train_errs.append(tr_err)
    test_errs.append(test_err)

#Plot the errors
plt.plot(x_plot, train_errs, label = 'Train error')
plt.plot(x_plot, test_errs, label = 'Test error')
plt.xlabel('Number of data points used')
plt.ylabel('Mean sq error')
plt.show()

#The test error stabilizes for aroubd 300 datapoints.
#it increases when we decrease the num of datapoints => no underfitting

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error

lambdas = [0.01,0.1,1,10,100]

#Exrend and split the data into two parts for validation and training
M_train= math.floor(N_train*0.8)
y_train = y[:M_train]
y_val = y[M_train:N_train]
Z=PolynomialFeatures(2).fit_transform(X)
Z_train=Z[:M_train]
Z_val=Z[M_train:N_train]
Z_test=Z[N_train:]


def get_errs(model, Z_train, y_train, Z_val, y_val):
    model.fit(Z_train, y_train)
    return (mean_squared_error(y_train, model.predict(Z_train)),
        mean_squared_error(y_val, model.predict(Z_val)))

#Calculate errors
errR=[]
errL=[]
for l in lambdas:
    errL.append(get_errs(Lasso(l), Z_train, y_train, Z_val, y_val)[0])
    errR.append(get_errs(Ridge(l), Z_train, y_train, Z_val, y_val)[0])


#Find optimal errs and print them
bestL = lambdas[np.argmin(errL)]
print("The best λ for Lasso is ", bestL, " with training and test error",
      get_errs(Lasso(bestL), Z[:N_train], y[:N_train], Z_test, y_test) )

bestR = lambdas[np.argmin(errR)]
print("The best λ for Ridge is ", bestR, " with training and test error",
      get_errs(Ridge(bestR), Z[:N_train], y[:N_train], Z_test, y_test) )






