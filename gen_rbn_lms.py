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
    # Radial-Basis Function
    ##############################################
    def rbf(self, x, mu, cov):
        # return np.exp(-1 * .5 * (1/np.power(sigma, 2)) * np.linalg.norm(x-mu))
        # cov = [[sigma[0], 0], [0, sigma[1]]]
        # output = ( np.exp( -.5 * np.dot(np.dot(( x - mu ).T, np.linalg.inv(cov)) , ( x - mu )) ) / 
        #             ( np.power( np.power(2*np.pi, len(x)) * np.linalg.det(cov) , .5) ) )
        output = ( np.exp( -.5 * np.dot(np.dot(( x - mu ).T, np.linalg.inv(cov)) , ( x - mu )) ) )
        
        return output



    ##############################################
    # Estimate class spread, sigma
    ##############################################
    def estimate_sigma(self, centers):
        # find dmax
        dmax = np.zeros((len(centers[0])))
        for i in range(len(centers[0])):
            for c1 in centers:
                for c2 in centers:
                    if(c1[i] - c2[i] > dmax[i]):
                        dmax[i] = c1[i] - c2[i]
        
        sigma = 1 * dmax / np.power( 2 * len(centers), .5 )

        return sigma



    ##############################################
    # Forward Pass
    ##############################################
    def forward(self, x):
        # for all J
        for i in range(len(self.centers)):
            self.iota[i] = self.rbf(x, self.centers[i], self.center_covs[i])
        out = np.dot( self.iota.T, self.hl_weights )

        return out


        
    ##############################################
    # Plot stuff
    ##############################################
    def plot_stuff(self, *args):
        fig, ax = plt.subplots()
        for arr in args:
            ax.scatter(arr[:,0], arr[:,1])
        plt.show()



    ##############################################
    # Train: k-means then LMS
    ##############################################
    def train(self, data, labels):

        # Kmeans to find centers
        km = kmeans.KMeans(self.k)
        self.centers = km(data, error_target=.001)

        # Estimate Covariances
        uniqs = list(set(sorted(labels)))
        self.covariances = {}
        for lab in uniqs:
            data_class = np.array([d for d,l in zip(data,labels) if l==lab])
            self.covariances[lab] = self.estimate_cov(np.transpose(data_class,[1,0]))
        self.center_covs = []
        for lab in labels:
            self.center_covs.append(self.covariances[lab])

        # Plot
        self.plot_stuff(data, self.centers)

        # Forward pass
        self.X = np.zeros((len(data), self.k))
        for i in range(len(data)):
            iota = np.zeros((self.k))

            # for all J
            for j in range(len(self.centers)):
                out = self.rbf(data[i], self.centers[j], self.center_covs[j])
                iota[j] = out
            
            self.X[i] = copy.deepcopy(iota)

        # Backward Pass
        self.hl_weights = np.dot( np.dot( np.linalg.inv(np.dot(self.X.T , self.X)) , self.X.T ) , labels )

        return self.centers
