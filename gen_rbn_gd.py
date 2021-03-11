import numpy as np
from tqdm import tqdm
import copy
import pickle
import time

import kmeans

import matplotlib.pyplot as plt
import pandas
from collections import Counter

##############################################
# RBF Network 
# One hidden layer
##############################################
class RBN:
    def __init__(self, k, outputs, config=None):
        # X  (1, 2)
        # H1 (2, 3)
        # W  (3, 3)
        # O  (3, 1)
        self.k = k
        self.o = outputs
        self.iota = np.zeros((self.k, self.o))

        if(config):
            with open(config, 'rb') as file:
                contents = pickle.load(file)
                self.centers = contents[0]
                self.hl_weights = contents[1]
                self.sigma = contents[2]
                print('Config Loaded!')


    ##############################################
    # Radial-Basis Function
    ##############################################
    def rbf(self, x, mu, sigma):
        output = np.exp(-1 * .5 * (1/np.power(sigma, 2)) * np.linalg.norm(x-mu))
        # cov = [[sigma[0], 0], [0, sigma[1]]]
        # output = ( np.exp( -.5 * np.dot(np.dot(( x - mu ).T, np.linalg.inv(cov)) , ( x - mu )) ) / 
        #             ( np.power( np.power(2*np.pi, len(x)) * np.linalg.det(cov) , .5) ) )
        # output = ( np.exp( -.5 * np.dot(np.dot(( x - mu ).T, np.linalg.inv(cov)) , ( x - mu )) ) )
        
        return output



    # Manually compute mean multivariate
    def mean_mv(self, data):
        means = []
        # d is dimension (not sample)
        for d in data:
            sum = 0.0
            for val in d:
                sum += val
            means.append(sum/len(d))

        return means


    # Estimate covariance matrix
    def estimate_cov(self, data):
        n = len(data[0])
        d = len(data)
        covs = np.zeros((d,d))
        means = self.mean_mv(data)

        # dimension first data (d,n)
        for x in range(d):
            for y in range(d):
                x_v = (data[x] - means[x])
                y_v = (data[y] - means[y])
                covs[x,y] = np.sum(x_v * y_v) / n

        return np.array(covs)


    ##############################################
    # Estimate class spread, sigma
    ##############################################
    def estimate_sigma(self, centers):
        # find dmax
        dims = 1 if len(centers.shape) == 1 else centers.shape[1]
        dmax = 0

        for c1 in centers:
            for c2 in centers:
                if(np.linalg.norm(c1 - c2) > dmax):
                    dmax = np.linalg.norm(c1 - c2)
        
        sigma = 1 * dmax / np.power( 2 * len(centers), .5 )

        return sigma


    ##############################################
    # Forward Pass
    ##############################################
    def forward(self, x):
        # for all J
        for i in range(len(self.centers)):
            self.iota[i] = self.rbf(x, self.centers[i], self.sigma[i])
        out = np.dot( self.iota.T, self.hl_weights )

        return out[0][0]



    ##############################################
    # Backward Pass
    ##############################################
    def backward(self, output, y, alpha=.01):

        # weights
        error = y - output
        np.expand_dims([error], axis=1)
        dE_w = -1 * np.dot(self.iota, error)
        self.hl_weights = self.hl_weights + (-1 * alpha * dE_w)

        return error
        

    def plot_stuff(self, *args):
        fig, ax = plt.subplots()
        for arr in args:
            ax.scatter(arr[:,0], arr[:,1])
        plt.show()


    ##############################################
    # Train: k-means then forward/backward
    ##############################################
    def train(self, data, labels, alpha=.01, epochs=100, batch_size=50, dw_target=.01, save_config=False):
        # hidden layer output
        self.iota = np.zeros((self.k, self.o))

        # weights with random numbers
        self.hl_weights  = np.random.uniform(-max(labels), max(labels), size=(self.k, self.o))

        # Init centers and covariances
        # Kmeans to find centers
        km = kmeans.KMeans(self.k)
        self.centers = km(data, error_target=.001)
        
        # # Estimate Covariances
        # uniqs = list(set(sorted(labels)))
        # self.covariances = {}
        # for lab in uniqs:
        #     data_class = np.array([d for d,l in zip(data,labels) if l==lab])
        #     self.covariances[lab] = self.estimate_cov(np.transpose(data_class,[1,0]))
        # self.center_covs = []
        # for lab in labels:
        #     self.center_covs.append(self.covariances[lab])

        # Estimate sigma
        sigma = self.estimate_sigma(self.centers)
        self.sigma = np.repeat(sigma, self.k)

        best_weights = None
        best_error   = np.inf

        # self.plot_stuff(data, self.centers)

        ## Epochs
        last_20 = []
        epochs_error = []
        for k in tqdm(range(epochs)):
            labels_ep = []
            outs_ep   = []
            old_h1 = copy.deepcopy(self.hl_weights)
            batch_error = []
            for b in range(batch_size):
                ind = np.random.randint(0, len(data))
                x = data[ind]
                y = labels[ind]
                
                # forward pass
                output = self.forward(x)

                labels_ep.append(y)
                outs_ep.append(output)

                # backward
                error = self.backward(output, y, alpha=alpha)
                batch_error.append( error )

            batch_avg_error = np.sum(batch_error) / batch_size

            if(abs(batch_avg_error) < best_error):
                best_error   = abs(batch_avg_error)
                best_weights = self.hl_weights

            epochs_error.append(batch_avg_error)
            # Break if below error target
            if(np.linalg.norm(self.hl_weights - old_h1) < dw_target):
                break

        self.hl_weights = best_weights

        if(save_config):
            self.save_config()
    
        print('Completed Training:',k+1,'epochs')
        print('Best Error:',best_error)
        return epochs_error, self.centers

    
    # Save out
    def save_config(self,):
        with open('models/config_'+str(time.time())+'.pkl', 'wb') as file:
            pickle.dump((self.centers, self.hl_weights, self.sigma), file)