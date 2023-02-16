#%%
import cvxpy as cp
import numpy as np
# %%
def rbf(x,y,gamma):
    return np.exp(-gamma*np.linalg.norm(x-y)**2)

def GaussianHK(x,y,s,t, sig, sigh):
    a = .5*(s+t)
    b = .5*(x+y)
    return rbf(x,y,sig)*rbf(s,t,sig)*rbf(a,b,sigh)

def pairs(X1,X2):
    out = []
    for x in X1:
        for y in X2:
            out.append([x,y])
    return(np.array(out))

def Hichol(HK, X):
    pivots = pairs(X,X)
    n = len(X)**2

    # initialize pivots, all diags equal to 1
    D = np.ones(n)
    G = np.empty((n,n))

    for k in range(n):
        G_k = np.zeros(n)

        G_k[k] = D[k]**.5
        
        for j_k in range(k,n):
            
            G_k[j_k] = HK(pivots[j_k,0,:], pivots[j_k,1,:],pivots[k,0,:],pivots[k,1,:],1,1)
            for j in range(k):
                G_k[j_k] -= G[j_k, j]*G[k, j]

            if G_k[k] != 0:
                G_k[j_k] /= G_k[k]

            D[j_k] = D[j_k] - G[j_k,k]**2

        G = np.hstack((G, G_k.reshape(-1,1)))

        if np.abs(D[k]) <= .0001:
            return G[:,n:]

    return G[:,n:]


# %%
K = Hichol(GaussianHK, X_train[:4,:])
print(K.shape)
print(K > 0)
# %%
