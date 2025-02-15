# Importing libraries
import numpy as np
import pandas as pd
import random

# Function to read input file
def ReadFile():
    return pd.read_csv('/Users/Sajeeth/Documents/Quant Portfolio/Machine Learning Projects/K-Means Clustering/Data.csv', encoding='utf-8', index_col=False)

# Function to write output file (data with an additional column representing which cluster each data point is in)
def WriteFile(df):
    return df.to_csv('/Users/Sajeeth/Documents/Quant Portfolio/Machine Learning Projects/K-Means Clustering/Clustered Data.csv', encoding='utf-8', index=False)

# Function to calculate the euclidean distance between 2 data points
def distance(p1,p2):
    return np.sqrt(np.sum((p1 - p2)**2))

# Call our file reading function
data = ReadFile()

def KMeansCluster(data,k):
    # Seed for reproducibility of results
    random.seed(0)
    
    # Initialise our array of centroids
    centroids = []
    # We will take our first centroid to be any random datapoint within our dataframe
    centroids.append(data.loc[random.randint(0,len(data)+1), :'col2'].to_numpy())
    
    # CHANGE INTO 2D ARRAY SINCE WE DO NOT NEED TO TABULATE/DISPLAY DISTANCES
    # We will make a dataframe to store distances of points from centroids so we can call from it instead of calculating each distance every time
    distances = pd.DataFrame()
    
    for a in range(k-1):
        # Initialise and store all distances from all points to a centroid
        d=[]
        for i in range(len(data)):
            d.append(distance(np.array(data.loc[i, :'col2']),centroids[a]))
        distances[f'd{a+1}'] = d
        
        # Here we isolate the case where we only have 1 predefined centroid since we do not need to take the minimum of multiple distances
        if a == 0:
            c = 1/(np.sum(distances['d1']**2))
            w = c*(distances['d1']**2)
            temp = int(random.choices(population=np.linspace(0,len(data)-1,len(data)),weights=w,k=1)[0])
            centroids.append(np.array(data.iloc[temp,:2]))
        
        # Here we deal with the case where we have multiple predefined centroids
        else:
            distances['mins'] = distances[list(distances.columns)].min(axis=1)
            c = 1/(np.sum(distances['mins']**2))
            w = c*(distances['mins']**2)
            temp = int(random.choices(population=np.linspace(0,len(data)-1,len(data)),weights=w,k=1)[0])
            centroids.append(np.array(data.iloc[temp,:2]))
            distances.drop(['mins'],axis=1,inplace=True)
    
    # We can quickly sort our centroids by their y-values
    centroids = sorted(centroids, reverse=True, key=lambda k: k[1])

    # Initialize clusters list
    clusters = [[] for _ in range(k)]
    
    # Loop until convergence
    # Initialise our current iteration and maximum iterative value
    max_iter = 50
    iteration = 0
    # Initialise our old centroids so we can check for equality in our while statement
    old_centroids = [[0]*2 for _ in range(k)]
    while np.any(np.not_equal(centroids,old_centroids)) and iteration < max_iter:

        # Clear previous clusters and initiliase array to store cluster values
        clusters = [[] for _ in range(k)]
        temptemp=[]
        
        # Assign each point to the closest centroid and store which cluster it belongs to
        for point in np.array(data.iloc[:,:2]):
            distances_to_each_centroid = [distance(point, centroid) for centroid in centroids]
            cluster_assignment = np.argmin(distances_to_each_centroid)
            temptemp.append(cluster_assignment)
            clusters[cluster_assignment].append(point)
        
        # Assign old centroids as centroids for comparison in our while statement
        old_centroids = centroids
        # Calculate new centroids by taking the mean of all points within a respective cluster
        new_centroids = [np.array(np.mean(cluster,axis=0)) for cluster in clusters]
        # Assign centroids as these new centroids
        centroids = new_centroids
        # Increase iteration by 1
        iteration += 1
    
    # Write the corresponding cluster for each datapoint in our dataframe
    data['cluster'] = temptemp
    # Call writing output file function
    WriteFile(data)
    
    return

KMeansCluster(data,4)