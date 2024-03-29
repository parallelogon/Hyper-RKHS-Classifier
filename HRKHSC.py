import numpy as np
from sklearn import svm
from scipy.sparse.linalg import cg
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class HRKHSC_2ST(BaseEstimator, ClassifierMixin):
    def __init__(self, r, lambda_Q_ = 1, intercept = True, C_ = 1, scale = "scale", subsample = None):
        self.lambda_Q_ = lambda_Q_
        self.r = r
        self.intercept = intercept
        self.Omega = None
        self.model = None
        self.C_ = C_
        self.scale = scale

        self.subsample = subsample


    def Z(self, Omega, X):
        return np.sqrt(2/self.r) * np.cos(Omega @ X.T)

    def pairs(self, X1, X2):
        n1, d = X1.shape
        n2 = X2.shape[0]
        return np.reshape(X1[:,None] - X2[None,:], (n1*n2, d))

    def D(self, Omega, X, X_prime = None):
        if X_prime is None:
            return self.Z(Omega, self.pairs(X,X))
        else:
            return self.Z(Omega, self.pairs(X, X_prime))

    def kernel_func(self, gamma, Omega):
        

        def k(X, Y):
            n1 = X.shape[0]
            n2 = Y.shape[0]
            return np.reshape(self.D(Omega, X, Y).T @ gamma, (n1,n2))

        return k


    def fit(self, X, y, verbose = False):
        X,y = check_X_y(X,y)
        self.classes_ = unique_labels(y)
        y = y.reshape(-1,1)

        n, dim = X.shape


        # randomly permute data
        I = np.random.permutation(n)
        X = X[I,:]
        y = y[I,:]


        # subsample for kernel learning

        if self.subsample is not None:
            assert self.subsample < 1.0
            I = np.random.choice(n,int(self.subsample*n), replace = False)

        Xt = X[I,:]
        yt = y[I,:]


        n = len(I)
        self.n = n

        yt = yt.reshape(-1,1)

        ones = np.ones((n,1))


        Xt = np.hstack((ones, Xt))

        ones1 = np.ones((len(X),1))
        X = np.hstack((ones1,X))

        self.dim = X.shape[-1]

        # roll random features (Omega) from mvnorm with intercept from uniform[0,2pi]
        if self.Omega is None:
            mu = np.zeros(dim)
            sig = np.eye(dim)
            Omega = np.random.multivariate_normal(mu, sig, size = self.r)
            uni = np.random.uniform(size = (self.r,1))*2*np.pi
            Omega = np.hstack((uni,Omega))
            self.Omega = Omega
        else:
            Omega = self.Omega

        # find the kernel matrix, k
        z = self.D(Omega, Xt)
        self.z = z

        # Woodbury Identity to solve for beta
        Y = np.reshape(yt @ yt.T, (-1,1))
        #Y = (Y+1)/2.
        T = Y.copy()
        Y = z @ Y
        Y = np.linalg.inv(self.lambda_Q_*np.eye(self.r) + z @ z.T) @ Y
        Y = z.T @ Y
        Y = T - Y
        self.beta = 1/self.lambda_Q_ * np.maximum(0,Y) # project onto positive orthant
        self.gamma = self.z @ self.beta
        self.model = svm.SVC(kernel = self.kernel_func(gamma = self.gamma, Omega = self.Omega), C = self.C_, gamma = self.scale)

        self.model.fit(X, y.ravel())

    def predict(self, X):
        n_test,dim = X.shape
        
        assert dim + 1 == self.dim, "Dimension of input incorrect"

        X = np.hstack((np.ones((n_test,1)),X))

        y_hat = self.model.predict(X)

        return np.reshape(y_hat, -1)



class HRKHSC_HD(BaseEstimator, ClassifierMixin):

    def __init__(self, r, lambda_ = 1, lambda_Q_ = 1, intercept = True, MAX_ITER = 10):
        self.lambda_ = lambda_
        self.lambda_Q_ = lambda_Q_
        self.MAX_ITER = MAX_ITER
        self.r = r
        self.intercept = intercept
        self.Omega = None


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

        # shuffle X and y, and include intercept

        n, dim = X.shape
        self.n = n

        I = np.random.permutation(self.n)
        X = X[I,:]
        y = y[I,:]

        y = y.reshape(-1,1)

        ones = np.ones((n,1))
        X = np.hstack((ones,X))

        self.X = X
        self.y = y


        # roll random features (Omega) from mvnorm with intercept from uniform[0,2pi]

        if self.Omega is None:
            mu = np.zeros(dim)
            sig = np.eye(dim)
            Omega = np.random.multivariate_normal(mu, sig, size = self.r)
            uni = np.random.uniform(size = (self.r,1))*2*np.pi
            Omega = np.hstack((uni,Omega))
            self.Omega = Omega
        else:
            Omega = self.Omega

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

        X = np.hstack((np.ones((n_test,1)),X))

        z_pred = self.D_pred(self.Omega, X, self.X)
        k = np.reshape(z_pred.T @ (self.z @ self.Beta), (n_test, self.n))
        y_hat = k @ self.Alpha + self.b > 0
        y_hat = 2*y_hat - 1

        return np.reshape(y_hat, -1)
    
    

class HRKHSC_LD(BaseEstimator, ClassifierMixin):

    def __init__(self, r,
                 lr = 0.01,
                 lambda_ = 1.,
                 lambda_Q_ = 1.,
                 MAX_ITER = 10,
                 N_LOOPS = 5,
                 scale = 1):
        
        self.lambda_ = lambda_
        self.lambda_Q_ = lambda_Q_
        self.MAX_ITER = MAX_ITER
        self.r = r
        self.lr = lr
        self.N_LOOPS = N_LOOPS
        self.Omega = None
        self.scale = scale

    def Z(self, Omega, X):
        r = Omega.shape[0]
        return np.sqrt(2/r) * np.cos(Omega @ X.T)
        

    def fit(self, X, y, verbose = False):

        X, y = check_X_y(X,y)
        self.classes_ = unique_labels(y)
        y = y.reshape(-1,1)

        # shuffle X and y, and include intercept

        n, dim = X.shape
        self.n = n

        I = np.random.permutation(self.n)
        X = X[I,:]
        y = y[I,:]

        y = y.reshape(-1,1)


        ones = np.ones((n,1))
        X = np.hstack((ones,X))

        self.X = X
        self.y = y

        
        # roll random features (Omega) from mvnorm with intercept from uniform[0,2pi]
        #mu = np.zeros(dim)
        #sig = np.eye(dim)
        #Omega = np.random.multivariate_normal(mu, sig, size = self.r)
        if self.Omega is None:
            Omega = np.random.randn(self.r, dim)
            Omega *= np.sqrt(self.scale)
            uni = np.random.uniform(size = (self.r,1))*2*np.pi
            Omega = np.hstack((uni,Omega))
            self.Omega = Omega
        else:
            Omega = self.Omega

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
                
                gamma = np.maximum(1e-10, gamma)
                #lr = self.lr/np.sqrt(1. + self.lr * (t + t_j))
                #t_j += 1
                
                #lr = self.lr/np.sqrt(1. + self.lr *(t* + t_i))
                #t_i += 1
                
            #lr = self.lr / np.sqrt(1. + self.lr * t*self.MAX_ITER)

                
            t += 1
        self.Alpha = alpha
        self.Gamma = gamma
        self.b = b
 

            
        return train_loss

    def predict(self, X):
        n_test,dim = X.shape
        assert dim + 1 == self.X.shape[1], "Dimension of input incorrect"

        X = np.hstack((np.ones((n_test,1)),X))
        X = X
        z_pred = self.Z(self.Omega, X).T

        y_hat = np.sign(z_pred @ (self.Alpha * np.sqrt(self.Gamma)) + self.b)


        return y_hat.reshape(len(y_hat))

