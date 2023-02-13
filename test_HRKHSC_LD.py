# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 13:48:55 2022

@author: Zachary Jones
"""

from HRKHSC import HRKHSC_LD as HRKHSC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_validate

n = 100
dim = 2
r = 10

# random data, two normal distributions with random cov matrices insure SPD
N = 10 #multiplies the mean and var by sqrt(N) and N for two heteroskedastic distr.
mu1 = np.zeros(dim)
mu2 = np.sqrt(N)*np.random.rand(dim)

cov1 = np.random.randn(dim,dim)
cov1 = cov1 - np.diag(np.diag(cov1)) + np.diag(np.random.rand(dim))
cov1 = .5*(cov1 + cov1.T)

while((np.linalg.eig(cov1)[0] < 0).any()):
    cov1 = cov1 + np.diag(np.random.rand(dim))

cov2 = N*np.random.randn(dim,dim)
cov2 = cov2 - np.diag(np.diag(cov2)) + N*np.diag(np.random.rand(dim))
cov2 = .5*(cov2 + cov2.T)

while((np.linalg.eig(cov2)[0] < 0).any()):
    cov2 = cov2 + N*np.diag(np.random.rand(dim))

X1 = np.random.multivariate_normal(mu1,cov1,size = n//2)
X2 = np.random.multivariate_normal(mu2, cov2, size = n//2)
X = np.vstack((X1, X2))

ones = np.ones((n//2,1))
y = np.vstack((ones, -1*ones)).ravel()

# split into train, val, test
n_train = np.floor(n*0.8).astype('int')
n_test = n - n_train

I = np.random.choice(n, n, replace = False)
X = X[I,:]
y = y[I]
X_train = X[:n_train,:]
y_train = y[:n_train]

# scale data here for easier plotting
Xmin = np.min(X_train, axis = 0)
Xmax = np.max(X_train, axis = 0)
X = (X - Xmin)/(Xmax - Xmin)

X_train = X[:n_train,:]

X_test = X[n_train:, :]
y_test = y[n_train:]


clf = HRKHSC(1, lambda_Q_ = .01, lambda_ = 1, lr = .01, MAX_ITER = 20)
clf.N_LOOPS = 5

'''
loss = clf.fit(X_train, y_train)

fig = plt.figure()
plt.plot(loss)
y_hat = clf.predict(X_test)
print(np.mean(y_test*y_hat > 0))


# make a color dictionary and select colors for plot
fig = plt.figure()
cdict = {-1: 'red', 1:'blue'}
colors = []
for yi in y_train:
    colors.append(cdict[yi])
    
plt.scatter(X_train[:,0],X_train[:,1],c= colors, label = 'Train')

colors = []
for yi in y_test:
    colors.append(cdict[yi])
    
plt.scatter(X_test[:,0],X_test[:,1],marker = 'v', c= colors, label = 'Test')


h = .02  # step size in the mesh
buffer = h

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - buffer, X[:, 0].max() + buffer
y_min, y_max = X[:, 1].min() - buffer, X[:, 1].max() + buffer

exes = np.arange(x_min, x_max, h)
whys = np.arange(y_min, y_max, h)
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                    np.arange(y_min, y_max, h))

# Plot the decision boundary
zz = np.c_[xx.ravel(), yy.ravel()]
zz = clf.predict(zz)
zz = zz.reshape(xx.shape)
plt.contour(xx, yy, zz, 0,cmap=plt.cm.Paired)
plt.legend()
plt.show()
'''
#params = {'lr':[0.1,1e-2,1e-3], 'lambda_':[1e-3,1e-1,1,100.0],'lambda_Q_':[1e-3,1e-1,1,100.0], 'r':[1,10,100]}

###############

params = {'lambda_':[1e-3,1e-2,1e-1,1.], 'r':[1, 10, 100,200]}#, 'lr':[.0001,.001,.01]}
clf_grid = GridSearchCV(clf, params, verbose = 3, cv = 5)
clf_grid.fit(X_train,y_train)
clf = clf_grid.best_estimator_

clf.MAX_ITER = 100
clf.N_LOOPS = 30

cvresults = cross_validate(clf, X, y, cv=5)
scores = 100*cvresults['test_score']
print("CVAL Test Score (std): %.2f%% (%.2f)" % (scores.mean(), scores.std()))
print('CVAL Max: %.2f' % scores.max())
# fit with all the data
loss = clf.fit(X_train,y_train)

print('train-test score: ', np.mean(y_test * clf.predict(X_test) > 0))
fig = plt.figure()
plt.plot(loss)
plt.show()
# make a color dictionary and select colors for plot
fig = plt.figure()
cdict = {-1: 'red', 1:'blue'}
colors = []
for yi in y_train:
    colors.append(cdict[yi])
    
plt.scatter(X_train[:,0],X_train[:,1],c= colors, label = 'Train')

colors = []
for yi in y_test:
    colors.append(cdict[yi])
    
plt.scatter(X_test[:,0],X_test[:,1],marker = 'v', c= colors, label = 'Test')

h = .02  # step size in the mesh
buffer = h

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - buffer, X[:, 0].max() + buffer
y_min, y_max = X[:, 1].min() - buffer, X[:, 1].max() + buffer

exes = np.arange(x_min, x_max, h)
whys = np.arange(y_min, y_max, h)
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                    np.arange(y_min, y_max, h))

# Plot the decision boundary
zz = np.c_[xx.ravel(), yy.ravel()]
zz = clf.predict(zz)
zz = zz.reshape(xx.shape)
plt.contour(xx, yy, zz, 0,cmap=plt.cm.Paired)
plt.legend()
plt.show()