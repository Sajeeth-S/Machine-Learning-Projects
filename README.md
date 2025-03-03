# **K-Means Clustering Algorithm**
This repository contains a K-Means Clustering Algorithm Visualisation project. Here we will:
1) Explain how this algorithm works step by step, and explain the maths behind it.
2) Go through some use cases and drawbacks when using it.
3) Implement the algorithm in Python from scratch with some test data.
4) Plot a Voronoi Diagram in order to easily visualise how the clustering has worked and to deem the effectiveness of it.
5) Work with actual DJIA and S&P500 stock data to cluster based on returns/volatility and RSI/ATR

## Why have I made this?
I started this project since I had been building an unsupervised trading algorithm, and in order to group assets based on similar features, I decided a K-Means Clustering approach would be best. However, in researching this algorithm online, I saw steps being labelled without much thought behind why they were done and many resources had turned a blind eye to the limitations and need for K-Means++ implementation or visualisation to truly understand how it worked. Thus, in order to achieve this, I have made this project.

## Optimisations/Features
- I have managed to implement a K-Means++ approach to initialising the first $k$ centroids. This is where the first centroid is chosen uniformly at random from the data points that are being clustered, after which each subsequent centroid is chosen from the remaining data points with probability proportional to its squared distance from the point's closest existing centroid.<br> Although this takes greater time to initialise, it significantly reduces the time to run the actual K-Means Clustering Algorithm. Hence overall, this is a optimisation in the long run.
- An issue with the Voronoi Diagram implementation through the SciPy library is that the regions are not corresponding to our centroids and clusters. I have managed to amend this, allowing for greater visualisation since we can easily colour the regions and data points within a cluster the same.

## What have I learnt from this?
Instead of simply importing libraries and running pre-existing functions, I have thorougly understood how this algorithm works and I feel I am able to incorporate this way of thinking into other machine learning methods.

I have also learnt the importance and use cases of clustering stocks for portfolio management.

Moreover, I have learnt about the existence of Voronoi Diagrams and how useful they can be in visualising data. To further this, I have come across a very interesting problem surrounding these diagrams: "In a Voronoi Diagram with $n$ uniformly chosen sites, what is the average perimeter of a random region in terms of $n$?".

## Improvements to be made
- I could store all iterations of centroids, such that I can plot a Voronoi Diagram for each iteration and iteratively see the change each time. For now, I have only plotted the start and end iterations. However, with the K-Means++ implementation, only a few iterations are required for most datasets.
- I have stored my distances in a dataframe when implementing K-Means++, in order to tabulate and display distances, however this is not necessary. So to save time and computation, I should instead store them in simple numpy arrays.
- I would like to implement other clustering methods that can deal with non-spherical data better and also avoid clustering spherically.
