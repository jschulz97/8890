import numpy as np
from matplotlib import pyplot as plt

# make data - using uniform dist and ranges
def generate_data(data_ranges, length, epsilon=0.0):
    # get max dims
    dims_x = [np.inf, -np.inf]
    dims_y = [np.inf, -np.inf]
    for clus in data_ranges:
        if(clus[0][0] < dims_x[0]):
            dims_x[0] = clus[0][0]
        if(clus[0][1] > dims_x[1]):
            dims_x[1] = clus[0][1]
        if(clus[1][0] < dims_y[0]):
            dims_y[0] = clus[1][0]
        if(clus[1][1] > dims_y[1]):
            dims_y[1] = clus[1][1]

    # generate
    data = np.zeros((1,2))
    labels = []
    for i,clus in enumerate(data_ranges):
        for j in range(length):
            if(np.random.rand() >= epsilon):
                data = np.append(data, [[
                    np.random.uniform(clus[0][0], clus[0][1]),
                    np.random.uniform(clus[1][0], clus[1][1]),
                ]], axis=0)
                labels.append(i)
            else:
                data = np.append(data, [[
                    np.random.uniform(dims_x[0], dims_x[1]),
                    np.random.uniform(dims_y[0], dims_y[1])
                ]], axis=0)
                labels.append(9)

    data = data[1:]

    return data, labels


# Split data for display - plt
def split_data(data, labels):
    data_sep = []
    uniq_labels = list(set(sorted(labels)))
    for cl in uniq_labels:
        data_class = np.zeros((1,2))
        for i,lab in enumerate(labels):
            if(cl == lab):
                data_class = np.append(data_class, [data[i]], axis=0)
        data_class = data_class[1:]
        data_sep.append(data_class)
    return data_sep, uniq_labels


# graph separated data w/ plt
def plt_graph_sep_data(data):
    fig, ax = plt.subplots()
    for cl in data:
        ax.scatter(cl[:,0], cl[:,1])
    plt.show()


def expand_labels(labels):
    uniqs = list(set(sorted(labels)))
    expanded = []
    for lab in labels:
        new_exp = [0] * len(uniqs)
        new_exp[uniqs.index(lab)] = 1
        expanded.append(new_exp)
    return expanded


