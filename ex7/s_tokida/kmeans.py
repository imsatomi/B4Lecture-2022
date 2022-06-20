import random
import numpy as np

def init_random(data, num_c):
    """define initial centroids randomly
    Args:
        data (ndarray): imput data
        num_c (int): number of clusters
    Returns:
        centroid (ndarray)
    """

    centroid = data[random.sample(range(data.shape[0]), num_c)]

    return centroid

def k_means_plusplus(data, num_c):
    """define initial centroids by kmeans++ algorithm
    Args:
        data (ndarray): imput data
        num_c (int): number of clusters
    Returns:
        centroid (ndarray)
    """
    
    centroid=np.zeros(data.shape[1] * num_c).reshape(num_c, data.shape[1])
    centroid[0, :] = data[np.random.choice(range(data.shape[0]), 1)]
    d = ((data - centroid[0]) ** 2).sum(axis = 1)
    p = d / d.sum()
    d_min = d

    for k in range(1, num_c):
        centroid[k, :] = data[np.random.choice(range(data.shape[0]), 1, p=p)]
        d = ((data - centroid[k]) ** 2).sum(axis = 1)
        d_min = np.minimum(d_min, d)
        p = d_min / d_min.sum()

    return centroid

def k_means(data, num_c, centroid):
    """clustering by kmeans method
    Args:
        data (ndarray): imput data
        num_c (int): number of clusters
    Returns:
        cluster (ndarray)
        centroid (ndarray) 
    """

    distance = np.zeros((num_c, data.shape[0]))
    cluster = np.zeros(data.shape[0])

    count = 0
    max_iter = 100
    
    while(count < max_iter):
        distance = np.sum((centroid[:,np.newaxis,:] - data[np.newaxis,:,:]) ** 2, axis = -1)

        distance = np.array([np.sum((centroid[i] - data) ** 2, axis = 1) for i in range(num_c)])
        cluster = np.argmin(distance, axis = 0)
        
        centroid_bef = centroid
        centroid = np.array([np.mean(data[cluster == i], axis = 0) for i in range(num_c)])
        count += 1
        if (centroid_bef == centroid).all():
            # print('count:', count)
            break
    
    return cluster, centroid