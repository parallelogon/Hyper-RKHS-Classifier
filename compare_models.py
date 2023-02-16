#%%
import numpy as np
from HRKHSC import HRKHSC_2ST, HRKHSC_HD, HRKHSC_LD
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_validate
from Data.Data import *
import pickle
#%%

#####
# Define a Datasest
#####
set = 'ionosphere'
with open('Data/' + set + '.pkl', 'rb') as inp:
    data = pickle.load(inp)

#data.X/=np.sqrt(2)
X_train,y_train,X_test,y_test = data.train_test(0.6)

y_train[y_train!= 1] = -1
y_test[y_test!=1] = -1

# %%
#####
# define a list of models and parameters
#####

def score(pred, true):
    return 100*np.mean(pred.reshape(true.shape) == true)

r0 = 256
models = {"HRKHS_2ST":\
    {"model": HRKHSC_2ST(r0),\
    "params": {#'r':[1,10,100],
    "C_":[100],
    "gamma_":["auto", "scale"]
    }},

    "HRKHS_HD": \
        {"model": HRKHSC_HD(r0),\
        "params":{'lambda_':[1/len(X_train)/100], #1e-3,1e-1,1,100],
        #'r':[1,10,100]
        }},
    
    "HRKHS_LD":\
        {"model":HRKHSC_LD(r0, MAX_ITER=100, N_LOOPS=5),\
            "params": {'lambda_':[1/len(X_train)/100], #[1e-3,1e-1,1.0],
             "lr":[0.1,0.01], 
             #'r':[1, 10, 100]
             }}
}
# %%

#####
# Cross Validate and fit
#####

for model in models.keys():
    clf = models[model]["model"]
    params= models[model]["params"]
    clf_grid = GridSearchCV(clf, params, verbose = 2, cv = 3)
    clf_grid.fit(X_train,y_train)
    clf = clf_grid.best_estimator_
    clf.fit(X_train,y_train)
    hold_out_score = score(clf.predict(X_test), y_test)
    train_score = score(clf.predict(X_train), y_train)
    models[model]["scores"] = {"train_score": train_score, "holdout_score": hold_out_score}

    #print("CVAL Test Score (std): %.2f%% (%.2f)" % (scores.mean(), scores.std()))
    #print('CVAL Max: %.2f' % scores.max())

    models[model]["model"] = clf
# %%
for model in models.keys():
    print(model, models[model]["model"].r, models[model]["scores"]["holdout_score"], models[model]["scores"]["train_score"])
# %%