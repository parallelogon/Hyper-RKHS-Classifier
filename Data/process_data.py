#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from Data.Data import *
import pickle 

#%%

def save(obj, name):
    with open(name, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


#%%
iris = pd.read_csv("iris.data", header = None)
category_dict = {name: num for name,num in zip(iris[4].unique(), [1,2,3])}

iris["y"] = iris[4].apply(lambda x: category_dict[x])

irisDat = Data(iris[[0,1,2,3]].to_numpy(), iris["y"].to_numpy())

save(irisDat, "iris.pkl")
# %%

ionosphere = pd.read_csv("ionosphere.data", header = None)

cat_dict = {name:num for name,num in zip(ionosphere[34].unique(), [e for e in range(len(ionosphere[34].unique()))])}

ionosphere["y"] = ionosphere[34].apply(lambda x: cat_dict[x])
ionosphere[0] = 2*ionosphere[0]-1

ionosphere.drop(1, axis = 1, inplace = True)
n_col = len(ionosphere.columns)
ionosphereDat = Data(ionosphere[[ionosphere.columns[e] for e in range(n_col-2)]].to_numpy(), ionosphere["y"].to_numpy())

save(ionosphereDat, "ionosphere.pkl")
# %%

sonar = pd.read_csv("sonar_csv.csv")
sonar["y"] = sonar.Class.astype("category").cat.codes

n_col = len(sonar.columns)
sonarDat = Data(sonar[[sonar.columns[e] for e in range(n_col-2)]].to_numpy(), sonar["y"].to_numpy())

save(sonarDat, "sonar.pkl")
# %%
import re
x = []
y = []
with open("seeds_dataset.txt") as seeds:
    lines = seeds.readlines()
    for line in lines:
        line_sep = re.split("\t+", line)#line.split("\t")
        y.append(int(line_sep[-1]))
        x.append([float(d) for d in line_sep[:-1]])

seedDat = Data(np.array(x), np.array(y))

save(seedDat, "seed.pkl")
# %%

bc = pd.read_csv("wdbc.data", header = None)

bc["y"] = bc[1].astype("category").cat.codes

x = bc[[i for i in range(2, len(bc.columns)-1)]]

wdbcDat = Data(x.to_numpy(), bc["y"].to_numpy())

save(wdbcDat, "breast_cancer.pkl")
# %%
