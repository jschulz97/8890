import numpy as np


class KNN():
    def __init__(self, k, train_x, train_y):
        self.k       = k
        self.train_x = train_x
        self.train_y = train_y
    

    def __call__(self, *args):
        return self.predict(*args)

    
    def forward(self, *args):
        return self.predict(*args)


    def predict(self, sample):
        # Get dist to every data point
        dists = []
        for x1 in self.train_x:
            dists.append(np.abs(np.linalg.norm(sample - x1)))
        dists = np.array(dists)

        # sort
        ind_sort = np.argsort(dists)

        # score
        scores = [0] * len(list(set(sorted(self.train_y))))
        for ki,i in zip(range(self.k), ind_sort):
            scores[self.train_y[i]] += 1

        # classify
        return scores.index(max(scores))
        