import numpy as np

class Data:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        return (self.X[idx,:], self.y[idx])

    def train_test(self, proportion, choices = None, standardize = False, normalize = False):

        assert proportion <= 1.0

        n = len(self.X)

        if choices is None:
            train = np.random.choice(n, int(n*proportion), replace=False)
        else:
            train = choices

        test = np.array([e for e in range(n) if e not in train])

        self.train_id = train
        self.test_id = test

        X_train = self.X[train,:]
        y_train = self.y[train]

        X_test = self.X[test,:]
        y_test = self.y[test]

        if standardize:
            mu = np.mean(X_train, axis = 0)
            std = np.std(X_train, axis = 0)

            X_train = (X_train - mu)/std
            X_test = (X_test - mu)/std

        if normalize:
            datamin = np.min(X_train, axis = 0)
            datamax = np.max(X_train, axis = 0)

            X_train = (X_train - datamin)/(datamax-datamin)
            X_test = (X_test - datamin)/(datamax-datamin)

        return X_train, y_train, X_test, y_test

