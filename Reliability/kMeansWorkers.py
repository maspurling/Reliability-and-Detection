# -*- coding: utf-8 -*-
"""
    Created on Fri Jun 11 09:58:54 2021
    
    Genereates correctness-confidence graph for a random sample of workers
    
    @author: Makenzie Spurling
"""

import random as rnd
import pandas as pd
import math
import matplotlib.pyplot as plt
import statistics as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class Worker:
    def __init__(self, _low=[], _up=[]):
        k = len(_low)
        self.mid = [(_low[i] + _up[i])/2 for i in range(k)] 
        self.rad = [(-_low[i]/2 + _up[i])/2 for i in range(k)]
        mid_c = st.mean(self.mid)
        rad_c = st.mean(self.rad)
        self.mean = [st.mean(_low), st.mean(_up)]
        self.center = sum(self.mean)/2
        self.conf = [abs(0.5 - self.mid[i]) + 0.5 - self.rad[i] for i in range(k)]
        '''if mid_c > 0.5:
            self.confidence = self.mean[0]
        else:
            self.confidence = 1 - self.mean[1]'''
        self.confidence = abs(self.center -0.5) + 0.5 -(self.mean[1]-self.mean[0])/2
        s = 0
        for i in range(k):
            s += abs((self.mid[i]-mid_c)*(self.rad[i] -rad_c))
        s = 2*s/(k-1)
        var = st.variance(self.mid, mid_c) + st.variance(self.rad, rad_c) + s
        self.stability = math.sqrt(var)
        pr = probability_x_unif(_low, _up) 
        self.entropy = entropy(pr)
    
        
def ivls(a, c=0.5, k=25):
    ''' ivls(a, c, k)
            Return k random intervals with midpoint near c
    '''
    b = rnd.uniform(0, 1)
    mid = [rnd.uniform(c-b, c+b) for i in range(k)]
    for i in range(k):
        if mid[i] > 1:
            mid[i] = 1
        if mid[i] < 0:
            mid[i] = 0
    rad = [min(mid[i], 1-mid[i]) for i in range(k)]
    
    if a < 80:        
        r =[rnd.uniform(0, rad[i]) for i in range(k)]
    else:
        r =[rnd.uniform(rad[i], 1) for i in range(k)]
    
    if a < 60:
        _low = [mid[i] - r[i] for i in range(k)]
        _up = [mid[i] + r[i] for i in range(k)]
    elif a < 75:
        _low = [mid[i] - (b+r[i]) for i in range(k)]
        _up = [mid[i] + r[i] for i in range(k)]
    elif a < 90:
        _low = [mid[i] - r[i] for i in range(k)]
        _up = [mid[i] + (b+r[i]) for i in range(k)]
    elif a < 100:
        mid = [rnd.uniform(c, c) for i in range(k)]
        _low = [mid[i] for i in range(k)]
        _up = [mid[i] for i in range(k)]

    for i in range(k):
        if _low[i] < 0:
            _low[i] = 0
        if _up[i] > 1:
            _up[i] = 1
        if _low[i] > _up[i]:
            _low[i], _up[i] = _up[i], _low[i]
    return _low, _up
       
def pdf_x_unif(x_low, x_up):
    ''' pdf_x_unif(x_low, x_up)
            Return pdf of an interval vector x assuming uniform 
            distribution for each interval x_i in x.
    '''  
    n = len(x_low)
    pdf_i = []
    #Assume uniform distribution
    for i in range(n):
        if x_up[i] - x_low[i] == 0:
            # print('Warning: constant interval detected')
            pdf_i.append(float('inf'))
        else:
            pdf_i.append(1/(x_up[i] - x_low[i]))

    # Partition to segment
    c = x_low + x_up # Adding low and up
    c.sort()
    # Initiate frequncy count for each segment as zero 
    seg = []
    for i in range(2*n-1):
        seg.append([c[i], c[i+1], 0])    

    # For each x_i in x locate its indices in segment list
    for i in range(n):
        l = c.index(x_low[i])
        u = c.index(x_up[i])
        # Accumulate segment pdf count
        for j in range(u-l):
            seg[l+j][2] += pdf_i[i] 
    for i in range(2*n-1):
        seg[i][2] /= n
        
    return seg # Segments with pdf

def probability_x_unif(x_low, x_up):
    ''' probability_x_unif(x_low, x_up)
            Return the probability list for all segments
    '''
    n = len(x_low)
    seg = pdf_x_unif(x_low, x_up)
    p = []
    
    for i in range(2*n -1):
        p.append(seg[i][2]*(seg[i][1]-seg[i][0]))
        
    return p

def entropy(p):
    ''' entropy(p)
            Returns entropy of interval probability
    '''
    e = 0
    for i in range(len(p)):
        if (p[i] != 0): # If a pdf is 0 ignore it
            e -= (p[i]*math.log(p[i]))
    return e

def silhouette(df):
    ''' silhouette(df)
            Graphs silhouette coefficient to find # of clusters
    '''
    kmeans_kwargs = {"init": "random", "n_init": 10,
                      "max_iter": 300, "random_state": 42,}  
    silhouette_coefficients = []
    for k in range(2, 20):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(df)
        score = silhouette_score(df, kmeans.labels_)
        silhouette_coefficients.append(score)
        
    plt.style.use("fivethirtyeight")
    plt.plot(range(2, 20), silhouette_coefficients)
    plt.xticks(range(2, 20))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Coefficient")
    plt.show()          

def sse(df):
    ''' sse(df)
            Graphs SSE to find # of clusters
    '''
    kmeans_kwargs = {"init": "random", "n_init": 10,
                      "max_iter": 300, "random_state": 42,}
    sse = []
    for k in range(1, 20):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)
    
    plt.style.use("fivethirtyeight")
    plt.plot(range(1, 20), sse)
    plt.xticks(range(1, 20))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.show()

def main(k=100):
    cor, con = [], []
    seeds = [rnd.uniform(0, 1) for i in range(k)] 

    # Get correctness & confidence of k workers
    for i in range(k):
        low, up = ivls(i, seeds[i], 35)
        w = Worker(low, up)
        cor.append(w.center)
        con.append(w.confidence)   

    # Put worker correctness & confidence into dataframe
    df = pd.DataFrame()
    df['x'] = cor
    df['y'] = con
    
    # Calculate # of clusters for kmeans
    # silhouette(df)
    # sse(df)
    
    # Plot workers via correctness-confidence graph in clusters
    kmeans = KMeans(n_clusters=3).fit(df)
    centroid = kmeans.cluster_centers_
    
    plt.scatter(df['x'], df['y'],
                c=kmeans.labels_.astype(float), s=30, marker= 'v', alpha=.9)
    plt.scatter(centroid[:, 0], centroid[:, 1], c='red', s=60, marker='x')
    
    # plt.scatter(df['x'], df['y']) # Plot w/o clusters
    plt.xlabel("Correctness", fontsize=15)
    plt.ylabel("Confidence", fontsize=15)
    plt.show()

if __name__ == '__main__':
    main()