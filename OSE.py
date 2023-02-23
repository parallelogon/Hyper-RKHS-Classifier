#%%
import numpy as np
from sklearn import svm
from scipy.optimize import nnls
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y
from sklearn.utils.multiclass import unique_labels
from scipy.sparse.linalg import cg

# %%

class OSE(BaseEstimator, ClassifierMixin):

    def __init__(self, r = None, rank = None, C_ = 100, lambda_Q_ = 1, prop = 1):
        self.C_ = C_
        self.lambda_Q_ = lambda_Q_
        self.r = r
        self.prop = prop
        self.rank = rank

    def num_pairs(self,n1,n2):
        return np.array([np.repeat([e for e in range(n1)], n2), np.tile([e for e in range(n2)], n1)]).T

    def hyper_sample(self, n, r):
        p = np.random.permutation(n)
        m = .5*(2*n - r + 1)*r

        pair = np.zeros((int(m),2))

        for i in range(r):
            for j in range(n):
                pair[int(0.5*(i)*(2*n-i+1) + (j-i)),0],pair[int(0.5*(i)*(2*n-i+1) + (j-i)),1] = p[i],p[j]

        return pair.astype(int), p[:r]


    def hyper_vec(self, K, pair):
        n = pair.shape[0]
        vecK = np.zeros((n,1))

        for i in range(n):
            vecK[i] = K[pair[i,0], pair[i,1]]
        
        return vecK
        
    def phyper_gaussian(self, X1,X2,pairs1,pairs2,sigma1, sigma2):
        x_,y_ = pairs1[:,0],pairs1[:,1]
        s_,t_ = pairs2[:,0],pairs2[:,1]

        xx = (X1[x_]+X1[y_])/2
        yy = (X2[s_]+X2[t_])/2

        a = np.exp(-.25/sigma1**2 *np.linalg.norm(X1[x_]-X1[y_], axis = -1)**2)
        b = np.exp(-.25/sigma1**2 *np.linalg.norm(X2[s_]-X2[t_], axis = -1)**2)
        c = np.exp(-.5/(sigma1**2 + sigma2**2) * np.linalg.norm(xx-yy, axis = -1)**2)

        return a*b*c

    def Hichol(self, X_train, pivots, sigma1, sigma2, rank = None, tol = 1e-3):

        n = len(pivots)
        D = np.zeros(n)
        G = np.zeros((n,1))

        J = set(range(n))
        I = list()



        D = np.zeros(n)

        for k in range(n):
            p1,p2 = pivots[k,:]
            D[k] = np.exp(-.25/sigma1*np.linalg.norm(X_train[p1,:]-X_train[p2,:])**2)

        G = np.zeros((n,1))

        if rank is None:
            end = n
        else:
            end = rank
            print('rank', rank)

        for k in range(end):
            i = np.argmax(D)

            I.append(i)
            J.remove(i)
            j = list(J)


            G[i,k] = np.sqrt(D[i])

            if k == 0:

                G[j,k] = 1.0/G[i,k] * self.phyper_gaussian(X_train,X_train,pivots[j,:],np.array([pivots[i,:]]), sigma1, sigma2)

                D[j] = D[j] - (G[j, k]**2).ravel()
                D[i] = 0

                if np.max(D) < tol:
                    return G[:,:k+1],I,J

                G = np.hstack((G, np.zeros((n,1))))

                continue
            else:

                G[j,k] = 1.0 / G[i, k] * (self.phyper_gaussian(X_train,X_train,pivots[j,:],np.array([pivots[i,:]]), sigma1, sigma2) - G[j, :].dot(G[i, :].T))
                G = np.hstack((G, np.zeros((n,1))))

                D[j] = D[j] - (G[j,k]**2).ravel()

                # eliminate selected pivot
                D[i] = 0


                # check residual lower bound and maximum rank
                if (np.max(D) < tol) and (rank is None):
                    return G[:,:k+1],I,J

        return G[:,:k+1],I,J

    def fit(self, X_train, y_train):


        X_train, y_train = check_X_y(X_train,y_train)
        self.classes_ = unique_labels(y_train)
        y_train = y_train.reshape(-1,1)

        # shuffle X and y, rescale X, and include intercept

        N, dim = X_train.shape
        self.N = N

        I = np.random.permutation(self.N)
        X_train = X_train[I,:]
        y_train = y_train[I,:]

        y_train = y_train.ravel()

        sigma1 = X_train.std()
        sigma2 = self.prop * sigma1

        N = len(X_train)

        if self.r is None:
            r = int(0.05*N)
        else:
            r = self.r

        sample, _ = self.hyper_sample(N, r)


        #X_train, sample, X_train.std(), X_train.std(), tol = .5)
        G,I,J = self.Hichol(X_train, sample.astype(int), sigma1, sigma2, rank = self.rank, tol = 1e-2)
        self.shape = G.shape


        Y = y_train.reshape(-1,1) @ y_train.reshape(-1,1).T

        V = self.hyper_vec(Y, sample.astype(int))
        V = (V+1.)/2.

        C = np.vstack([G@G.T, np.sqrt(self.lambda_Q_)*G.T])
        d = np.vstack([V, np.zeros((G.shape[-1],1))])

        beta,_ = nnls(C, d.ravel())
        I = beta.nonzero()
        sample = sample[I]
        beta = beta[I]
        self.beta = beta
        k_num = len(beta)

        pairY = np.zeros((k_num,1))
        meanY = np.zeros((k_num, X_train.shape[-1]))
        for s in range(k_num):
            Y = [X_train[sample[s,0],:], X_train[sample[s,1],:]]
            pairY[s] = np.exp(-.25/sigma1**2*np.linalg.norm(Y[0]-Y[1])**2)
            meanY[s,:] = 0.5*(Y[0]+Y[1])

        K_train = np.zeros((N,N))
        for i in range(N):
            for j in range(i+1):
                X = [X_train[i,:], X_train[j,:]]
                pair1 = np.exp(-.25/sigma1**2*np.linalg.norm(X[0]-X[1])**2)
                mean1 = .5*(X[0]+X[1])
                pairX = pair1*pairY
                k_xy = np.zeros((k_num, 1))
                for s in range(k_num):
                    pair3 = np.exp(-.5/(sigma1**2+sigma2**2)*np.linalg.norm(mean1-meanY[s,:])**2)
                    k_xy[s] = pairX[s]*pair3
                K_train[i,j] = k_xy.T.dot(beta)
                K_train[j,i] = K_train[i,j]

        clf = svm.SVC(kernel = "precomputed", C = self.C_)
        clf.fit(K_train, y_train.ravel())

        self.model = clf
        self.K_train = K_train
        self.k_num = k_num
        self.pairY = pairY
        self.meanY = meanY
        self.N = N
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.X_train = X_train


    def predict(self, X_test):

        N = self.N
        clf = self.model
        k_num = self.k_num
        pairY = self.pairY
        beta = self.beta
        meanY = self.meanY
        sigma1 = self.sigma1
        sigma2 = self.sigma2
        X_train = self.X_train

        #X_test = (X_test - self.Xmin)/(self.Xmax - self.Xmin)
        #X_test = (X_test - self.mu)/self.sig


        K_test = np.zeros((N, len(X_test)))
        for i in range(N):
            for j in range(len(X_test)):
                X = [X_train[i,:], X_test[j,:]]
                pair1 = np.exp(-.25/sigma1**2 * np.linalg.norm(X[0]-X[1])**2)
                mean1 = .5*(X[0]+X[1])
                pairX = pair1*pairY
                k_xy = np.zeros((k_num,1))
                for s in range(k_num):
                    pair3 = np.exp(-.5/(sigma1**2 + sigma2**2) * np.linalg.norm(mean1 - meanY[s,:])**2)
                    k_xy[s] = pairX[s]*pair3
                K_test[i,j] = k_xy.T.dot(beta)

        y_hat = clf.predict(K_test.T)

        return y_hat
# %%

"""def rbf(self, x,y,sig):

        if len(x.shape)==1:
            d = len(x)
        else:
            d = x.shape[-1]
        
        return (1/2/np.pi/sig**2)**(d/2)*np.exp(-np.linalg.norm(x-y)/2/sig**2)


    def hyper_gaussian(self, x,y,s,t,sigma1,sigma2):
        
        a = self.rbf(x,y, np.sqrt(2)*sigma1)
        b = self.rbf(s,t, np.sqrt(2)*sigma1)
        c = self.rbf((x+y)/2, (s+t)/2, np.sqrt(sigma1**2 + sigma2**2))

        return a*b*c

    def Hichol(self, X, sigma1, sigma2, pivots, tol=1e-3, rank = None):

        n = pivots.shape[0]
        print("n",n)
        D = np.zeros(n)
        G = np.zeros((n,1))

        J = set(range(n))
        I = list()

        G = np.empty((n,1))

        for k in range(n):
            p1,p2 = pivots[k,:]
            D[k] = np.exp(-.25/sigma1*np.linalg.norm(X[p1,:]-X[p2,:])**2)

        if rank is None:
            end = n
        else:
            end = rank
            print('rank', rank)

        for k in range(min(n,end)):
            i = np.argmax(D)
            I.append(i)

            try:
                J.remove(i)
            except:
                print(i)
                print(D)
                print(J)
            j = list(J)


            s_i,t_i = pivots[i,:]
            s,t = X[s_i,:], X[t_i,:]

            G_k = np.zeros((n,1))

            G_k[i] = np.sqrt(D[i])

            if k == 0:
                HKV = np.zeros((len(j),1))
                for ii,j_i in enumerate(j):
                    HKV[ii] = self.hyper_gaussian(X[pivots[j_i,0],:], X[pivots[j_i,1],:], s, t, sigma1, sigma2)

                G_k[j] = 1.0/G_k[i] *(HKV)
                G = G_k.copy()
                D[j] = D[j] - (G_k[j]**2).ravel()
                D[i] = 0
                if np.max(D) < 1e-3:
                    break

                continue
            else:
                HKV = np.zeros((len(j),1))
                for ii,j_i in enumerate(j):
                    HKV[ii] = self.hyper_gaussian(X[pivots[j_i,0],:], X[pivots[j_i,1],:], s,t, sigma1, sigma2)
                G_k[j] = 1.0 / G_k[i] * (HKV - np.reshape(G[j, :].dot(G[i, :].T),(-1,1)))


            D[j] = D[j] - (G_k[j]**2).ravel()

            # eliminate selected pivot
            D[i] = 0.

            G = np.hstack((G, G_k))


            # check residual lower bound and maximum rank
            if (np.max(D) < tol):
                if rank is None:
                    break

        return G,I,J"""