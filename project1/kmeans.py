import numpy as np
import copy


##############################################
# Simple K-Means
# Returns: (k,d) centers of k clusters
##############################################
class KMeans:
    def __init__(self, k):
        self.k      = k


    def __call__(self, data, eta=.1, error_target=.01):
        # Init random centers
        rand_indices = np.random.randint(0, len(data), size=(self.k,))
        centers = []
        for ind in rand_indices:
            centers.append(data[ind])
        centers = np.array(centers)

        # Update loop
        count = 0
        while(True):
            # Sample
            x = data[np.random.randint(0, len(data))]

            # Find best center
            mini = np.inf
            kx   = None
            for i in range(len(centers)):
                if(np.linalg.norm(x - centers[i]) < mini):
                    mini = np.linalg.norm(x - centers[i])
                    kx   = i
            
            # Update
            old_centers = copy.deepcopy(centers)
            centers[kx] = centers[kx] + ( eta * (x - centers[kx]) )

            # Break condition
            if(np.linalg.norm(centers - old_centers) < error_target):
                break

            count += 1
        # print('Num Updates:',count)

        return centers
