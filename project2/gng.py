import numpy as np

class GNG:
    def __init__(self, ):
        pass
    
    
    def __call__(self, ):
        pass
    
    
    def train(self, train_data, epochs=10):
        # Edges and Edge Age
        edges_C = np.zeros((len(train_data), len(train_data)))
        edges_age = np.zeros((len(train_data), len(train_data)))

        # 0. pick two neurons, farthest apart
        max_dist = 0
        furthest_points = ()
        for x1 in train_data:
            for x2 in train_data:
                dist = np.abs(np.linalg.norm(x1-x2))
                if(dist > max_dist):
                    max_dist = dist
                    furthest_points = (x1, x2)

        w_a, w_b = *furthest_points

        while(epochs > 0):
            epochs -= 1
            
            # 1. Pick input signal P(B)
            choice_i = np.random.randint(0, len(train_data))
            input_B = train_data[choice_i]

            # 2. Find two nearest units
            min_dist_1 = np.inf
            min_dist_2 = np.inf
            s_1 = None
            s_2 = None
            for x1 in train_data:
                dist = np.abs(np.linalg.norm(x1 - input_B))
                if(dist < min_dist_1):
                    s_2 = s_1
                    min_dist_2 = min_dist_1

                    min_dist_1 = dist
                    s_1 = x1

                elif(dist < min_dist_2):
                    min_dist_2 = dist
                    s_2 = x1

            print('2. Find two nearest points: min_dist_1:',min_dist_1,'min_dist_2:',min_dist_2)

            # 3. Increment age of all edges from s_1
            pass