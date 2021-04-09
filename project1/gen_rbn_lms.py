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
        self.k = k
        self.o = outputs

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
        output =  np.exp(-1 * .5 * (1/np.power(sigma, 2)) * np.linalg.norm(x-mu))
        # cov = [[sigma[0], 0], [0, sigma[1]]]
        # output = ( np.exp( -.5 * np.dot(np.dot(( x - mu ).T, np.linalg.inv(cov)) , ( x - mu )) ) / 
        #             ( np.power( np.power(2*np.pi, len(x)) * np.linalg.det(cov) , .5) ) )
        # output = ( np.exp( -.5 * np.dot(np.dot(( x - mu ).T, np.linalg.inv(cov)) , ( x - mu )) ) )
        
        return output



    ##############################################
    # Estimate class spread, sigma
    ##############################################
    def estimate_sigma(self, centers):
        # find dmax
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
        iota = np.zeros((self.k))
        for i in range(len(self.centers)):
            iota[i] = self.rbf(x, self.centers[i], self.sigma[i])
        out = np.dot( iota.T, self.hl_weights )

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
        self.centers = np.array(self.centers)

        # Estimate sigma
        sigma = self.estimate_sigma(self.centers)
        self.sigma = np.repeat(sigma, self.k)

        # Plot
        # self.plot_stuff(data, self.centers)

        # Forward pass
        self.X = np.zeros((len(data), self.k))
        for i in range(len(data)):
            iota = np.zeros((self.k))

            # for all J
            for j in range(len(self.centers)):
                out = self.rbf(data[i], self.centers[j], self.sigma[j])
                iota[j] = out
            
            self.X[i] = copy.deepcopy(iota)

        # Backward Pass
        self.hl_weights = np.dot( np.dot( np.linalg.inv(np.dot(self.X.T , self.X)) , self.X.T ) , labels )

        return self.centers
