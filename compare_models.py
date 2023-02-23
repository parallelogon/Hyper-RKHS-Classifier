#%%
import numpy as np
from HRKHSC import HRKHSC_2ST, HRKHSC_HD, HRKHSC_LD
from OSE import OSE
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from Data.Data import *
import pickle
#%%

#####
# Define a Datasest
#####


props = [.8]
dataset = 'seed'
iter_data = 5
iter_rf = 5
accs = {p:{i:{j:{"SVM":[], "HRKHS_LD":[], "HRKHS_2ST":[]}for j in range(iter_rf)} for i in range(iter_data)} for p in props}

names = ["HRKHS_2ST", "SVM"]
r0 = 200

def score(pred, true):
    return 100*np.mean(pred.reshape(true.shape) == true)

def iqr(X):
    lower, upper = np.percentile(X, [.25,.75])

    if upper-lower == 0:
        return X.std()**2

    return(upper - lower)

for p in props:

    for iter in range(iter_data):
        with open('Data/' + dataset + '.pkl', 'rb') as inp:
            data = pickle.load(inp)

        X_train,y_train,X_test,y_test = data.train_test(p, normalize = True)

        y_train[y_train!= 1] = -1
        y_test[y_test!=1] = -1


        #####
        # define a list of models and parameters
        #####

 

        for s in range(iter_rf):
            models = {"HRKHS_2ST":\
                {"model": HRKHSC_2ST(r0, C_ = 100, subsample=.2, scale = 1/iqr(X_train)),\
                "params": {#"C_": [1,10,100], #'r':[1,10,100],
                #"gamma_":["auto", "scale"]
                }},

                #"HRKHS_HD": \
                #    {"model": HRKHSC_HD(r0, 1/len(, X_train)/100),\
                #    "params":{#'lambda_':[1/len(X_train)/100], #1e-3,1e-1,1,100],
                    #'r':[1,10,100]
                #    }},
                
                "HRKHS_LD":\
                    {"model":HRKHSC_LD(r0, MAX_ITER=100,lr = .05, N_LOOPS=10, scale = 1, lambda_ = 1/len(X_train)/100),\
                        "params": {#[1e-3,1e-1,1.0],
                        "lr":[0.1, .05], 
                        #'r':[1, 10, 100]
                            }},

                #"OSE":\
                #    {"model":OSE(C_ = 100, r = int(.05*len(X_train))),
                #    "params": {
                #    }},

                "SVM":\
                    {"model":svm.SVC(C=100),
                    "params":{"C":[1,10,100],
                    "gamma": ["auto", "scale"]
                    }}
            }

            #####
            # Cross Validate and fit
            #####

            for model in models.keys():
                clf = models[model]["model"]
                params= models[model]["params"]
                print(model)
                I = np.random.choice(len(X_train), int(len(X_train)*.2),replace = False)

                if len(params)!= 0:
                    clf_grid = GridSearchCV(clf, params, verbose = 2, cv = 2)
                    clf_grid.fit(X_train[I,:],y_train[I])
                    clf = clf_grid.best_estimator_
                else:
                    clf = models[model]["model"]

                clf.fit(X_train,y_train)
                hold_out_score = score(clf.predict(X_test), y_test)
                train_score = score(clf.predict(X_train), y_train)
                models[model]["scores"] = {"train_score": train_score, "holdout_score": hold_out_score}
                models[model]["model"] = clf

            local_accs = {}
            for model in models.keys():
                print(model, models[model]["scores"]["holdout_score"], models[model]["scores"]["train_score"])
                local_accs[model] = models[model]["scores"]["holdout_score"]

                accs[p][iter][s][model].append(models[model]["scores"]["holdout_score"])

# %%
#dim = models["OSE"]["model"].shape[-1]
#print(dim, np.log2(dim))
#%%
np.save("./results/accs3_" + dataset + "_" + str(r0), accs)
# %%
