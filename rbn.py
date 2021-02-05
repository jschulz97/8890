import numpy as np
from tqdm import tqdm
import copy

import kmeans

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
        output = ( np.exp( -.5 * np.dot(np.dot(( x - mu ).T, np.linalg.inv(cov)) , ( x - mu )) ) / 
                    ( np.power( np.power(2*np.pi, len(x)) * np.linalg.det(cov) , .5) ) )
        
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
        
        sigma = dmax / np.power( 2 * len(centers), .5 )

        return sigma



    ##############################################
    # Forward Pass
    ##############################################
    def forward(self, x):
        # for all J
        for i in range(len(self.centers)):
            self.iota[i] = self.rbf(x, self.centers[i], self.sigma)
        # for all O
        self.iota = self.rbf(x, self.centers, self.sigma)
        out = np.dot( np.expand_dims(self.iota, axis=0), self.hl_weights )

        return out




    ##############################################
    # Backward Pass
    ##############################################
    def backward(self, output, y, alpha=.01):
        error = y - output
        big_e = np.sum( [ np.power(e, 2) for e in error ] ) / 2.0
        
        






    ##############################################
    # Train: k-means then forward/backward
    ##############################################
    def train(self, data, labels, alpha=.01, epochs=100, batch_size=50, error_target=.01):
        # hidden layer
        self.iota = np.zeros((self.k, self.o))

        # weights with random numbers
        self.hl_weights  = np.random.normal(0, 1, size=(self.k, self.o))

        # init errors
        error = np.array((self.k, self.o))

        # Init centers and covariances
        km = kmeans.KMeans(self.k)
        self.centers = km(data, error_target=.001)
        self.sigma = self.estimate_sigma(self.centers)

        # Expand labels
        soft_labels = self.make_soft_labels(labels)

        ## Epochs
        for k in tqdm(range(epochs)):
            old_h1 = copy.deepcopy(self.hl_weights)
            for b in range(batch_size):
                ind = np.random.randint(0, len(data))
                x = data[ind]
                y = soft_labels[ind]
                
                # forward pass
                output = self.forward(x)

                # backward
                self.backward(output, y, alpha=alpha)


            
            
            # Break if below error target
            if(np.linalg.norm(self.hl_weights - old_h1) < error_target):
                break
        
        print('Completed Training:',k,'epochs')



    def make_soft_labels(self, labels):
        soft_labels = []
        uniqs = list(set(sorted(labels)))

        for lab in labels:
            new_label = [0] * len(uniqs)
            new_label[lab] = 1
            soft_labels.append(np.array(new_label))
        soft_labels = np.array(soft_labels)

        return soft_labels



    ###############################################
    # Testing
    def test(self, test_dim=0, rand=True, ):
        #Find test_dim
        if(test_dim == 0):
            test_dim = self.test_dim
        elif(test_dim > self.test_dim):
            print("\n\nWarning! Number of testing images ("+str(test_dim)+") too large. Increase test_dim parameter (currently "+str(self.test_dim)+") on object initialization.")
            test_dim = self.test_dim
        print("\nTesting on",test_dim,"images...") 
        self.test_dim = test_dim

        self.pred_res  = np.zeros((test_dim,10))   
        self.pred_ind  = np.zeros((test_dim,))

        v1  = np.ones((1,101))
        o   = np.ones((1,10))
        ind = []

        ## Input Data
        for l in progressbar(range(test_dim)):
            # Can decide with 'rand' parameter to test in order
            if(rand):
                # Get random index
                i = np.random.randint(low=0, high=self.test_dim, )
                while(i in ind):
                    i = np.random.randint(low=0, high=self.test_dim, )
                ind.append(i)
            else:
                i = l

            x = self.test_data[i]

            ## Forward pass
            #   (1,100)          (1,197)          (197,1)    
            for j in range(100): 
                v1[0][j] = np.dot(np.append(x,1), np.transpose(self.h1_weights[j]))
                v1[0][j] = self.actfx(v1[0][j])

            #   (1,10)          (1,101)  (101,1)    
            for j in range(10):
                o[0][j] = np.dot(v1,     np.transpose(self.out_weights[j]))
                o[0][j] = o[0][j]
            
            self.pred_res[l] = o[0]
            self.pred_ind[l] = i

        self.did_i_test = True
        self.classify()


    #############################
    # Classifier
    def classify(self, ):
        #print(self.pred_res[:10])

        for i in range(self.test_dim):
            mxi = np.argmax(self.pred_res[i])
            self.pred_res[i]      = np.zeros((10,))
            self.pred_res[i][mxi] = 1


    #############################
    # Plot error over updates
    def plot_error(self, show=True):  
        if(self.did_i_train):
            fig = plt.figure(figsize=(11,9))
            plt.plot(self.err_exp)
            plt.ylabel('error')
            plt.xlabel('updates')
            print("\nDisplaying error plot...\n")
            plt.savefig(self.cwd+'/error_plot_'+self.desc+'.jpg')
            if(show):    
                plt.show()
            plt.close()
        else:
            print('\nTrain the network first!\n')


    #############################
    # Plot deltas over updates
    def plot_deltas(self, show=True):  
        if(self.did_i_train):
            fig = plt.figure(figsize=(11,9))
            plt.plot(self.h1_delta_full.ravel())
            plt.ylabel('deltas')
            plt.xlabel('updates')
            print("\nDisplaying delta_h1 plot...\n")
            plt.savefig(self.cwd+'/deltah1_plot_'+self.desc+'.jpg',bbox_inches='tight')
            if(show):    
                plt.show()
            plt.close()
            fig = plt.figure(figsize=(11,9))
            plt.plot(self.ow_delta_full.ravel())
            plt.ylabel('deltas')
            plt.xlabel('updates')
            print("\nDisplaying delta_ow plot...\n")
            plt.savefig(self.cwd+'/deltaow_plot_'+self.desc+'.jpg',bbox_inches='tight')
            if(show):
                plt.show()
            plt.close()
        else:
            print('\nTrain the network first!\n')

    
    #############################
    # Show confusion matrix
    def show_cm(self, show=True):
        if(self.did_i_test):
            cm = np.zeros((10,10))

            good = 0
            for i in range(self.test_dim):
                ind = int(self.pred_ind[i])
                tpi = np.argmax(self.test_labels[ind])
                cm[tpi] += self.pred_res[i]

                #score
                if(tpi == np.argmax(self.pred_res[i])):
                    good += 1

            mx = np.max(cm)

            fig = plt.figure(figsize=(9,9))
            fig.suptitle('')

            ax = fig.add_subplot()
            im = ax.imshow(cm,vmax=mx,vmin=0)

            for j in range(10):
                for k in range(10):
                    text = ax.text(k,j,cm[j,k],ha="center", va="center", color="w",fontsize=10)

            plt.colorbar(im)
            plt.ylabel('Actual')
            plt.xlabel('Prediction')
            plt.xticks(np.arange(0, 10, 1))
            plt.yticks(np.arange(0, 10, 1))
            ax.set_ylim(10-0.5, -0.5)
            print("\nDisplaying confusion matrix...\n")
            plt.savefig(self.cwd+'/cm_mat_'+self.desc+'.jpg')
            if(show):    
                plt.show()
            plt.close()
            
            return (good/self.test_dim)
        else:
            print('\nTest on the network first!')
            return 0

    
    #############################
    # Show an image of one neuron's learned weights
    def show_weights(self, i=0, show=True):
        if(self.did_i_train):
            fig = plt.figure(figsize=(9,9))
            weights = self.h1_weights[i][:-1]
            weights = np.reshape(weights,(14,14))
            print("\nDisplaying learned weights...\n")
            plt.imshow(weights)
            plt.savefig(self.cwd+'/weights_img_'+self.desc+'.jpg')
            if(show):
                plt.show()
            plt.close()
        else:
            print('\nTrain the network first!\n')