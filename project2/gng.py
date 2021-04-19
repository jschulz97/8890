import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from utils import *

# class Edge:
#     def __init__(self, node_from, node_to):
#         self.nodes = (node_from, node_to)
#         self.age = 0

#     def isconnected(self, node):
#         return node in self.nodes


# class Graph:
#     def __init__(self,):
#         self.edges = []


#     def add_edge(self, node_from, node_to):
#         # Check if exists already
#         for i,edge in enumerate(self.edges):
#             if(edge.nodes == (node_from, node_to) or edge.nodes == (node_to, node_from)):
#                 print('Edge already exists, resetting age')
#                 self.edges[i].age = 0
#                 return False
        
#         # Add it to graph
#         self.edges.append(Edge(node_from, node_to))
#         return True


#     def del_edge(self,):
#         pass


#     def kill(self, age):
#         new_edges = []
#         for edge in self.edges:
#             if(edge.age <= age):
#                 new_edges.append(edge)

#         self.edges = new_edges

    
#     def get_nodes(self,):
#         nodes = []
#         for edge in self.edges:
#             if(edge.nodes[0] not in nodes):
#                 nodes.append(edge.nodes[0])
#             if(edge.nodes[1] not in nodes):
#                 nodes.append(edge.nodes[1])
        
#         return nodes


class GNG:
    def __init__(self, ):
        pass
    
    
    def __call__(self, ):
        pass
    
    
    def train(self, train_data, 
                epochs=10, 
                learning_b=.1, 
                learning_n=.001, 
                age_max=50, 
                gamma=5,
                error_alpha=.1,
                error_d=.9,
                error_target=False):
        # Edges and Edge Age
        self.edges      = np.zeros((2,2))
        self.edges_age  = np.zeros((2,2))
        self.error      = np.zeros((2))

        debug('0. pick two neurons')
        # 0. pick two neurons, farthest apart
        max_dist = 0
        furthest_points = ()
        for x1 in tqdm(train_data):
            for x2 in train_data:
                dist = np.abs(np.linalg.norm(x1-x2))
                if(dist > max_dist):
                    max_dist = dist
                    furthest_points = (x1, x2)

        w_a, w_b = furthest_points[0], furthest_points[1]
        self.neurons = np.append([w_a], [w_b], axis=0)
        self.edges[0][1] = 1

        epoch_error = []

        debug('GNG Loop')
        for e in tqdm(range(epochs)):
            
            #############################################
            # 1. Pick input signal P(B)
            debug('1. Pick input signal P(B)')
            choice_i = np.random.randint(0, len(train_data))
            input_B = train_data[choice_i]


            #############################################
            # 2. Find two nearest units
            debug('2. Find two nearest units')
            min_dist_1 = np.inf
            min_dist_2 = np.inf
            s_1 = None
            s_2 = None
            s_1_i = None
            s_2_i = None
            for i,x1 in enumerate(self.neurons):
                dist = np.abs(np.linalg.norm(x1 - input_B))
                if(dist < min_dist_1 and dist != 0.0):
                    s_2   = s_1
                    s_2_i = s_1_i
                    min_dist_2 = min_dist_1

                    min_dist_1 = dist
                    s_1   = x1
                    s_1_i = i

                elif(dist < min_dist_2 and dist != 0.0):
                    min_dist_2 = dist
                    s_2   = x1
                    s_2_i = i

            debug('Two nearest points: min_dist_1:',min_dist_1,'min_dist_2:',min_dist_2)


            #############################################
            # 3. Increment age of all edges from s_1
            debug('3. Incremement Age of all edges from s_1')
            for i,edge in enumerate(self.edges[s_1_i]):
                if(edge != 0):
                    self.edges_age[s_1_i][i] += 1
            

            #############################################
            # 4. Add error of signal to neighbor
            debug('4. Add error to neuron')
            self.error[s_1_i] += np.power(np.linalg.norm(s_1 - input_B), 2)


            #############################################
            # 5. Move s1 and neighbors towards B
            debug('5. Move s1 and neighbors towards B')
            self.neurons[s_1_i] = self.neurons[s_1_i] + learning_b * (input_B - s_1)

            # Move neighbors
            for i,edge in enumerate(self.edges[s_1_i]):
                if(edge == 1):
                    self.neurons[i] = self.neurons[i] + learning_n * (input_B - self.neurons[i])


            #############################################
            # 6. reset s1 and s2 age and edge
            debug('6. reset s1 and s2 age')
            self.edges[s_1_i][s_2_i] = 1
            self.edges_age[s_1_i][s_2_i] = 0


            #############################################
            # 7. remove old edges & unconnected neurons
            debug('7. remove old edges and neurons')
            for i in range(len(self.edges_age)):
                for j,age in enumerate(self.edges_age[i]):
                    if(age > age_max):
                        self.edges[i][j] = 0
                        self.edges_age[i][j] = 0

            # remove unconnected neurons
            is_connected = [0] * len(self.neurons)
            for i in range(len(self.neurons)):
                for j,conn in enumerate(self.edges[i]):
                    if(conn == 1):
                        is_connected[i] = 1
                        is_connected[j] = 1
 
            new_len = np.sum(is_connected)
            if(new_len != len(is_connected)):
                # iterate backwards to retain indices
                for i,conn in reversed(list(enumerate(is_connected))):
                    if(conn == 0):
                        self.neurons   = np.delete(self.neurons,   i, axis=0)
                        self.edges     = np.delete(self.edges,     i, axis=0)
                        self.edges     = np.delete(self.edges,     i, axis=1)
                        self.edges_age = np.delete(self.edges_age, i, axis=0)
                        self.edges_age = np.delete(self.edges_age, i, axis=1)



            #############################################
            # 8. Insert new neuron
            debug('8. Insert new neuron')
            if(e % gamma == 0 and e != 0):
                argmax = np.argmax(self.error)

                # get neighbor with largest error
                neighbors_error = []
                for i,edge in enumerate(self.edges[argmax]):
                    if(edge == 1):
                        neighbors_error.append(self.error[i])
                    else:
                        neighbors_error.append(0)
                neighbor_argmax = np.argmax(neighbors_error)

                w_Q_i = argmax
                w_Q = self.neurons[w_Q_i]
                w_f_i = neighbor_argmax
                w_f = self.neurons[w_f_i]

                # New neuron
                w_r = .5 * (w_Q + w_f)

                # expand arrays
                self.edges = np.append(self.edges, np.zeros((len(self.edges[0]), 1)), axis=1)
                self.edges = np.append(self.edges, np.zeros((1, len(self.edges[0]))), axis=0)

                self.edges_age = np.append(self.edges_age, np.zeros((len(self.edges_age[0]), 1)), axis=1)
                self.edges_age = np.append(self.edges_age, np.zeros((1, len(self.edges_age[0]))), axis=0)

                self.neurons = np.append(self.neurons, [w_r], axis=0)

                self.error[w_Q_i] = self.error[w_Q_i] * error_alpha
                err = self.error[w_f_i] = self.error[w_f_i] * error_alpha
                self.error = np.append(self.error, [err], axis=0)

                # Fix edges
                self.edges[w_Q_i][w_f_i] = 0
                self.edges_age[w_Q_i][w_f_i] = 0
                self.edges[w_f_i][w_Q_i] = 0
                self.edges_age[w_f_i][w_Q_i] = 0

                self.edges[w_Q_i][-1] = 1
                self.edges[-1][w_f_i] = 1

            

            #############################################
            # 9. Decrease error
            debug('9. Decrease system error')
            self.error = self.error * error_d


            #############################################
            # Plot!
            debug()
            debug()
            debug('Attemping to plot...')

            fig, ax = plt.subplots()
            ax.scatter(train_data[:,0], train_data[:,1])
            ax.scatter(self.neurons[:,0], self.neurons[:,1], color='orange')
            for i,_ in enumerate(self.edges):
                for j,edge in enumerate(self.edges[i]):
                    if(edge == 1):
                        ax.plot(self.neurons[[i,j],0], self.neurons[[i,j], 1], color='orange')

            plt.savefig('./output/'+str(e).zfill(4)+'.jpg')
            plt.close()

            # Save epoch error
            epoch_error.append(np.max(self.error))

            # Get out if error target is specified and met
            if(error_target and np.max(self.error) < error_target):
                print('Error target met!')
                break

        return self.neurons, epoch_error
