import numpy as np
from tqdm import tqdm
import copy

import kmeans

import matplotlib.pyplot as plt
import pandas
from collections import Counter

##############################################
# RBF Network 
# One hidden layer
##############################################
class RBN:
    def __init__(self, k, outputs):
        # X  (1, 2)
        # H1 (2, 3)
        # W  (3, 3)
        # O  (3, 1)
        self.k = k
        self.o = outputs


    ##############################################
    # Radial-Basis Function
    ##############################################
    def rbf(self, x, mu, sigma):
        # return np.exp(-1 * .5 * (1/np.power(sigma, 2)) * np.linalg.norm(x-mu))
        cov = [[sigma[0], 0], [0, sigma[1]]]
        # output = ( np.exp( -.5 * np.dot(np.dot(( x - mu ).T, np.linalg.inv(cov)) , ( x - mu )) ) / 
        #             ( np.power( np.power(2*np.pi, len(x)) * np.linalg.det(cov) , .5) ) )
        output = ( np.exp( -.5 * np.dot(np.dot(( x - mu ).T, np.linalg.inv(cov)) , ( x - mu )) ) )
        
        return output



    ##############################################
    # RBF Criterion Function (k-means equiv)
    ##############################################
    def criterion(self, X):
        sum = 0.0



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
            self.iota[i] = self.rbf(x, self.centers[i], self.sigma)
        out = np.dot( self.iota.T, self.hl_weights )

        return out



    ##############################################
    # Backward Pass
    ##############################################
    def backward(self, output, y, alpha=.01):
        error = y - output
        # big_e = np.sum( [ np.power(e, 2) for e in error ] ) / 2.00
        dE_w = -1 * np.dot(self.iota, error)
        # dE_w = -1 * error
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
    def train(self, data, labels, alpha=.01, epochs=100, batch_size=50, dw_target=.01):
        # hidden layer output
        self.iota = np.zeros((self.k, self.o))

        # weights with random numbers
        self.hl_weights  = np.random.uniform(-max(labels), max(labels), size=(self.k, self.o))

        # Init centers and covariances
        # Kmeans to find centers
        km = kmeans.KMeans(self.k)
        self.centers = km(data, error_target=.001)
        self.sigma = self.estimate_sigma(self.centers)

        # Expand labels
        # soft_labels = self.make_soft_labels(labels)

        best_weights = None
        best_error   = np.inf

        self.plot_stuff(data, self.centers)

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
                outs_ep.append(output[0][0])

                # backward
                error = self.backward(output, y, alpha=alpha)
                batch_error.append( error )

            batch_avg_error = np.sum(batch_error) / batch_size

            # # faux momentum?
            # last_20.append(abs(batch_avg_error))
            # while(len(last_20) > 20):
            #     last_20.pop(0)
            # if(len(last_20) == 20 and abs(np.average(last_20) - last_20[-1]) < .0001):
            #     alpha = alpha / 3.0
            #     print('new alpha:', alpha)
            #     last_20 = []

            if(abs(batch_avg_error) < best_error):
                best_error   = abs(batch_avg_error)
                best_weights = self.hl_weights

            epochs_error.append(batch_avg_error)
            # Break if below error target
            if(np.linalg.norm(self.hl_weights - old_h1) < dw_target):
                break

        self.hl_weights = best_weights

        # # dist of outputs/labels
        # # fixed bin size
        # print('Hist of iota')
        # new_iota = [i for i in self.iota if i > 1e-5] # take out zeros
        # print(len(new_iota))
        # bins = np.arange(-3, 3, .01) # fixed bin size
        # plt.xlim([min(new_iota)-1, max(new_iota)+1])
        # plt.hist(new_iota, bins=bins, alpha=0.5)
        # plt.show()

        # print('Hist of outputs')
        # bins = np.arange(-5, 5, .1) # fixed bin size
        # plt.xlim([min(outs_ep)-1, max(outs_ep)+1])
        # plt.hist(outs_ep, bins=bins, alpha=0.5)
        # plt.show()

        # # fixed bin size
        # print('Hist of weights')
        # bins = np.arange(-15, 15, .1) # fixed bin size
        # plt.xlim([min(self.hl_weights)-1, max(self.hl_weights)+1])
        # plt.hist(self.hl_weights, bins=bins, alpha=0.5)
        # plt.show()
        
        print('Completed Training:',k+1,'epochs')
        print('Best Error:',best_error)
        return epochs_error, self.centers



    def make_soft_labels(self, labels):
        soft_labels = []
        uniqs = list(set(sorted(labels)))

        for lab in labels:
            new_label = [0] * len(uniqs)
            new_label[lab] = 1
            soft_labels.append(np.array(new_label))
        soft_labels = np.array(soft_labels)

        return soft_labels


    # #############################
    # # Show confusion matrix
    # def show_cm(self, show=True):
    #     if(self.did_i_test):
    #         cm = np.zeros((10,10))

    #         good = 0
    #         for i in range(self.test_dim):
    #             ind = int(self.pred_ind[i])
    #             tpi = np.argmax(self.test_labels[ind])
    #             cm[tpi] += self.pred_res[i]

    #             #score
    #             if(tpi == np.argmax(self.pred_res[i])):
    #                 good += 1

    #         mx = np.max(cm)

    #         fig = plt.figure(figsize=(9,9))
    #         fig.suptitle('')

    #         ax = fig.add_subplot()
    #         im = ax.imshow(cm,vmax=mx,vmin=0)

    #         for j in range(10):
    #             for k in range(10):
    #                 text = ax.text(k,j,cm[j,k],ha="center", va="center", color="w",fontsize=10)

    #         plt.colorbar(im)
    #         plt.ylabel('Actual')
    #         plt.xlabel('Prediction')
    #         plt.xticks(np.arange(0, 10, 1))
    #         plt.yticks(np.arange(0, 10, 1))
    #         ax.set_ylim(10-0.5, -0.5)
    #         print("\nDisplaying confusion matrix...\n")
    #         plt.savefig(self.cwd+'/cm_mat_'+self.desc+'.jpg')
    #         if(show):    
    #             plt.show()
    #         plt.close()
            
    #         return (good/self.test_dim)
    #     else:
    #         print('\nTest on the network first!')
    #         return 0