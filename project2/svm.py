import numpy as np
from sklearn.utils import shuffle


class SVM:
    def __init__(self, learning_rate=.000001, reg_strength=10000):
        self.learning_rate = learning_rate
        self.reg_strength = reg_strength
        self.learned_weights = []
        self.learned_weights_flag = False


    def learn(self, X, y):
        # train the model
        print("Training started...")
        X = np.append(X, np.ones((1,X.shape[1])), axis=0)
        y = np.append(y, np.array([1]))
        W = self.sgd(X, y)
        print("Training finished.")
        print("weights are: {}".format(W))
        return W


    def cost(self, W, X, Y):
        # calculate hinge loss
        N = X.shape[0]
        distances = 1 - Y * (np.dot(X, W))
        distances[distances < 0] = 0  # equivalent to max(0, distance)
        hinge_loss = self.reg_strength * (np.sum(distances) / N)
        
        # calculate cost
        cost = 1 / 2 * np.dot(W, W) + hinge_loss
        return cost


    def calculate_cost_gradient(self, W, X_batch, Y_batch):
        # if only one example is passed (eg. in case of SGD)
        if type(Y_batch) != np.ndarray:
            Y_batch = np.array([Y_batch])
            X_batch = np.array([X_batch])
        distance = 1 - (Y_batch * np.dot(X_batch, W))
        dw = np.zeros(len(W))
        for ind, d in enumerate(distance):
            if max(0, d) == 0:
                di = W
            else:
                di = W - (self.reg_strength * Y_batch[ind] * X_batch[ind])
            dw += di
        dw = dw/len(Y_batch)  # average
        return dw


    def sgd(self, features, outputs):
        max_epochs = 5000
        weights = np.zeros(features.shape[1])
        nth = 0
        prev_cost = float("inf")
        cost_threshold = 0.01  # in percent
        # stochastic gradient descent
        for epoch in range(1, max_epochs):
            # shuffle to prevent repeating update cycles
            X, Y = shuffle(features, outputs)
            for ind, x in enumerate(X):
                ascent = self.calculate_cost_gradient(weights, x, Y[ind])
                weights = weights - (self.learning_rate * ascent)
            # convergence check on 2^nth epoch
            if epoch == 2 ** nth or epoch == max_epochs - 1:
                cost = self.cost(weights, features, outputs)
                print('Epoch:', str(epoch).rjust(4), '| Cost:', cost)
                # stoppage criterion
                if abs(prev_cost - cost) < cost_threshold * prev_cost:
                    self.learned_weights = weights
                    self.learned_weights_flag = True
                    return weights
                prev_cost = cost
                nth += 1
        self.learned_weights = weights
        self.learned_weights_flag = True
        return weights


    def predict(self, data, labels):
        if(not self.learned_weights_flag):
            print('\nError! Fit classifier with learn() before calling predict()\n')
            return
        
        data_new = data * self.learned_weights

        score = 0.0
        for x,y in zip(data_new, labels):
            if(np.dot(self.learned_weights.T , x) > 0.0):
                pred = 1.0
            elif(np.dot(self.learned_weights.T , x) < 0.0):
                pred = -1.0
            else: 
                pred = 0.0
            
            if(pred == y):
                score += 1.0

        print('\nPrediction Score:', str(int(score))+'/'+str(len(labels)), '=', score/len(labels))