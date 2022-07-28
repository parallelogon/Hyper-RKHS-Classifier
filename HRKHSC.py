import numpy as np
from scipy.sparse.linalg import cg
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import cvxpy as cp
class HRKHSC_HD(BaseEstimator, ClassifierMixin):

    def __init__(self, r, lambda_ = 1, lambda_Q_ = 1, intercept = True, MAX_ITER = 10):
        self.lambda_ = lambda_
        self.lambda_Q_ = lambda_Q_
        self.MAX_ITER = MAX_ITER
        self.r = r
        self.intercept = intercept

    def Z(self, Omega, X):
        r = Omega.shape[0]
        return np.sqrt(2/r) * np.cos(Omega @ X.T)
        
    def pairs(self, X1,X2):
        n1,d = X1.shape
        n2 = X2.shape[0]
        return np.reshape(X1[:,None] - X2[None,:], (n1*n2,d))

    def D(self, Omega, X):
        return self.Z(Omega, self.pairs(X,X))

    def D_pred(self, Omega, X1, X2):
        return self.Z(Omega, self.pairs(X1, X2))

    def fit(self, X, y, verbose = False):

        X, y = check_X_y(X,y)
        self.classes_ = unique_labels(y)
        y = y.reshape(-1,1)

        # shuffle X and y, rescale X, and include intercept

        n, dim = X.shape
        self.n = n

        I = np.random.permutation(self.n)
        X = X[I,:]
        y = y[I,:]

        y = y.reshape(-1,1)

        Xmax = np.max(X, axis = 0)
        Xmin = np.min(X, axis = 0)
        X = (X - Xmin)/(Xmax - Xmin)
        ones = np.ones((n,1))
        X = np.hstack((ones,X))

        self.X = X
        self.y = y

        self.Xmax = Xmax
        self.Xmin = Xmin

        # roll random features (Omega) from mvnorm with intercept from uniform[0,2pi]
        mu = np.zeros(dim)
        sig = np.eye(dim)
        Omega = np.random.multivariate_normal(mu, sig, size = self.r)
        uni = np.random.uniform(size = (self.r,1))*2*np.pi
        Omega = np.hstack((uni,Omega))
        self.Omega = Omega

        # Beta vector multiplies with the hyperkernel K and has dim (n**2 x 1)
        Beta = np.random.rand(n**2, 1)

        # find the kernel matrix, k
        z = self.D(Omega, X)
        self.z = z
        upper = np.vstack((0,ones)).T

        for iter in range(self.MAX_ITER):
            gamma = z @ Beta
            k = np.reshape(z.T @ gamma,(n,n))

            # construct and solve the linear system of equations Cx = d using
            # conjugate gradient descent.  C is low rank and should converge well
            # solution is in lower dimensional space than rank C

            M = 1/self.lambda_ * k + np.eye(n)

            if self.intercept == True:
                lower = np.hstack((ones,M))

                C = np.vstack((upper, lower))
                d = np.vstack((0,y))

                x,_ = cg(C,d)
                
                b = x[0]
                mu = x[1:].reshape(-1,1)

            else:
                C = M
                d = y
                x,_ = cg(C,d)

                b = 0
                mu = x.reshape(-1,1)


            # the Beta vector is the solution to the minimization problem for
            # the LSSVM lagrangian dual with a hyper-RKHS regularizer added
            # require Beta >= 0 in order to guarantee SPD kernel function
            Beta = 1/2/self.lambda_/self.lambda_Q_ * (mu * mu.T).reshape(-1,1)
            Beta = np.maximum(0, Beta)
            
            if verbose:
                Alpha = 1/self.lambda_ * mu
                y_hat = k @ Alpha + b
                train = np.mean(y_hat*y > 0)
                print("Iteration %d | Training Score: %.2f" % ((iter + 1), train))



        # solve for Alpha and calculate training loss
        Alpha = 1/self.lambda_* mu

        self.Beta = Beta
        self.Alpha = Alpha
        self.b = b

        
        return self

    def predict(self, X):
        n_test,dim = X.shape
        
        assert dim + 1 == self.X.shape[1], "Dimension of input incorrect"

        X = (X - self.Xmin)/(self.Xmax - self.Xmin)
        X = np.hstack((np.ones((n_test,1)),X))

        z_pred = self.D_pred(self.Omega, X, self.X)
        k = np.reshape(z_pred.T @ (self.z @ self.Beta), (n_test, self.n))
        y_hat = k @ self.Alpha + self.b > 0
        y_hat = 2*y_hat - 1

        return np.reshape(y_hat, -1)
    
    

class HRKHSC_LD(BaseEstimator, ClassifierMixin):

    def __init__(self, r,
                 lr = 0.001,
                 lambda_ = 1.,
                 lambda_Q_ = 1.,
                 MAX_ITER = 10):
        
        self.lambda_ = lambda_
        self.lambda_Q_ = lambda_Q_
        self.MAX_ITER = MAX_ITER
        self.r = r
        self.lr = lr
        self.N_LOOPS = 5

    def Z(self, Omega, X):
        r = Omega.shape[0]
        return np.sqrt(2/r) * np.cos(Omega @ X.T)
        

    def fit(self, X, y, verbose = False):

        X, y = check_X_y(X,y)
        self.classes_ = unique_labels(y)
        y = y.reshape(-1,1)

        # shuffle X and y, rescale X, and include intercept

        n, dim = X.shape
        self.n = n

        I = np.random.permutation(self.n)
        X = X[I,:]
        y = y[I,:]

        y = y.reshape(-1,1)

        Xmax = np.max(X, axis = 0)
        Xmin = np.min(X, axis = 0)
        X = (X - Xmin)/(Xmax - Xmin)
        ones = np.ones((n,1))
        X = np.hstack((ones,X))

        self.X = X
        self.y = y

        self.Xmax = Xmax
        self.Xmin = Xmin

        
        # roll random features (Omega) from mvnorm with intercept from uniform[0,2pi]
        #mu = np.zeros(dim)
        #sig = np.eye(dim)
        #Omega = np.random.multivariate_normal(mu, sig, size = self.r)
        Omega = np.random.randn(self.r, dim)
        uni = np.random.uniform(size = (self.r,1))*2*np.pi
        Omega = np.hstack((uni,Omega))
        self.Omega = Omega

        # low dim gamma vector is sum of kernel params, alpha is weights
        #gamma = np.random.rand(self.r)
        
#        gamma = np.random.rand(self.r, 1)
        alpha = np.random.randn(self.r, 1)
        gamma = np.ones_like(alpha)
        b = 0
        
        lr = self.lr
        train_loss = []
        
        t = 0
        gradA = 0
        gradG = 0
        beta = 0.99
        
        for iteration in range(self.N_LOOPS):
            '''
            having constant learning rate with RMSProp gamma seems to
            lead to stable and flexible classification boundaries
            
            decreasing learning rate with no RMSProp works too
            
            normalized gradient 
            
            
            '''
            t_i = 0
            #lr = self.lr / np.sqrt(1. + self.MAX_ITER * t)
             
            for iteration_alpha in range(self.MAX_ITER):
                z = self.Z(Omega, X).T
                y_hat = z @ (alpha * np.sqrt(gamma)) + b
                
                losses = np.maximum(0, 1 - y * y_hat)
                losses = losses.reshape(len(losses))
                loss = losses.sum()
                train_loss.append(loss)
                
                grad = -100 * y * z * np.sqrt(gamma).T
                grad[losses <= 0] = 0
                grad = grad.mean(axis = 0).reshape(-1,1)
                grad += self.lambda_ * alpha
                
                gradA = beta * grad**2 + (1 - beta) * gradA
                alpha -= lr * (1/np.sqrt(1 + gradA)) * grad
                
                grad_b = - 100 * y
                grad_b[losses <= 0] = 0
                grad_b = np.mean(grad_b)
                
                b -= lr * grad_b
                
                #lr = self.lr/np.sqrt(1. + self.lr *(t* + t_i))
                #t_i += 1
                
            #lr = self.lr / np.sqrt(1. + self.lr * t*self.MAX_ITER)
            t_j = 0 
            #lr = self.lr
            for iteration_gamma in range(self.MAX_ITER):
                z = self.Z(Omega, X).T
                y_hat = z @ (alpha * np.sqrt(gamma)) + b
                
                losses = np.maximum(0, 1 - y * y_hat)
                losses = losses.reshape(len(losses))
                loss = losses.sum()
                train_loss.append(loss)
                
                #sel = (gamma != 0).reshape(len(gamma))
                grad = -50 * y * z * alpha.T / np.sqrt(gamma).T
                grad[losses <= 0] = 0
                grad = grad.mean(axis = 0).reshape(-1,1)
                #grad[~sel] = 0
                #grad /= np.linalg.norm(grad)
                grad += self.lambda_Q_ * gamma
                
                gradG = (1 - beta) * gradG + beta * grad**2
                gamma -= lr * (1/np.sqrt(1 + gradG)) * grad
                
                gamma = np.maximum(1e-30, gamma)
                #lr = self.lr/np.sqrt(1. + self.lr * (t + t_j))
                #t_j += 1
                

                
                
            t += 1
        self.Alpha = alpha
        self.Gamma = gamma
        self.b = b
 

            
        return train_loss

    def predict(self, X):
        n_test,dim = X.shape
        assert dim + 1 == self.X.shape[1], "Dimension of input incorrect"

        X = (X - self.Xmin)/(self.Xmax - self.Xmin)
        X = np.hstack((np.ones((n_test,1)),X))

        z_pred = self.Z(self.Omega, X).T
        #y_hat = ((self.Alpha * np.sqrt(self.Gamma)).T @ z_pred) + self.b > 0
        #y_hat = self.Alpha @ z_pred >= 0
        y_hat = np.sign(z_pred @ (self.Alpha * np.sqrt(self.Gamma)) + self.b)
        #y_hat = 2*y_hat - 1

        return y_hat.reshape(len(y_hat))