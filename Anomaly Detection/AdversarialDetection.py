# -*- coding: utf-8 -*-
"""
    Anomaly detection on interval-valued workers using t- and
    F-tests
    
    Detects anomalous workers and returns correctness-confidence
    graph of anomalies
    
    @author: Makenzie Spurling
"""
import pandas as pd
import random as rnd
import numpy as np
import math
import statistics as st
from scipy.stats import t, f
import matplotlib.pyplot as plt

class Worker:
    def __init__(self, _low=[], _up=[]):
        k = len(_low)
        self.mid = [(_low[i] + _up[i])/2 for i in range(k)]
        #self.rad = [(_up[i] - _low[i])/2 for i in range(k)]
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
        self.var = st.variance(self.mid, mid_c) + st.variance(self.rad, rad_c) + s
        self.stability = math.sqrt(self.var)
        pr = probability_x_unif(_low, _up) 
        self.entropy = entropy(pr)
        
class Measure:
    def __init__(self, conMat):
        try:
            self.recall = conMat[0]/(conMat[0]+conMat[3])
        except:
            self.recall = float('inf')
        try:
            self.precision = conMat[0]/(conMat[0]+conMat[1]) 
        except:
            self.precision = float('inf')
        try:
            self.accuracy = (conMat[0]+conMat[2])/(conMat[0]+conMat[1]+conMat[2]+conMat[3])
        except:
            self.accuracy = float('inf')
        try:
            self.fScore = 2 * self.recall * self.precision/(self.recall + self.precision)
        except:
            self.fScore = float('inf')
            
def pdf_x_unif(x_low, x_up, work=-1):
    ''' pdf_x_unif(x_low, x_up)
            Return pdf of an interval vector x assuming uniform 
            distribution for each interval x_i in x
    '''  
    n = len(x_low)
    pdf_i = []
    #Assume uniform distribution
    for i in range(n):
        if x_up[i] - x_low[i] == 0:
            #print('Warning: constant interval detected')
            pdf_i.append(float('inf'))
        else:
            pdf_i.append(1/(x_up[i] - x_low[i]))

    if work != -1:
        cor = 0
        for i in range(n):
            cor += work[i][0]
    # Partition to segment
    c = x_low + x_up #adding low and up
    c.sort()
    # initiate frequncy count for each segment as zero 
    seg = []
    for i in range(2*n-1):
        seg.append([c[i], c[i+1], 0])    

    # for each x_i in x locate its indices in segment list
    for i in range(n):
        l = c.index(x_low[i])
        u = c.index(x_up[i])
        # accumulate segment pdf count
        for j in range(u-l):
            if work != -1:
                seg[l+j][2] += pdf_i[i] * work[i][0]
            else:
                seg[l+j][2] += pdf_i[i] 
    for i in range(2*n-1):
        if work != -1:
            seg[i][2] /= cor
        else:
            seg[i][2] /= n
        
    return seg # segments with pdf

def probability_x_unif(x_low, x_up, work=-1):
    ''' probability_x_unif(x_low, x_up)
            Return the probability list for all segments
    '''
    n = len(x_low)
    seg = pdf_x_unif(x_low, x_up, work)
    p = []
    
    for i in range(2*n -1):
        p.append(seg[i][2]*(seg[i][1]-seg[i][0]))
    return p

def entropy(p):
    ''' entropy(p)
            Return the entropy of interval probability
    '''
    e = 0
    for i in range(len(p)):
        if (p[i] != 0): # If a pdf is 0 ignore it
            e -= (p[i]*math.log(p[i]))
    return e

def ivls(a, c=0.5, k=25):
    ''' ivls(c, k)
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

def goldLabels(k, w):
    ''' labels(k, w)
            Return k labels on questions for a single worker
    '''
    r = [rnd.uniform(0, .1) for i in range(len(k))]
    low = [w[0] - r[i] for i in range(len(k))]
    up = [w[0] + r[i] for i in range(len(k))]
    for i in range(len(k)):
        if low[i] > up[i]:
            low[i], up[i] = up[i], low[i]
        if low[i] < 0:
            low[i] = 0
        if up[i] > 1:
            up[i] = 1
            
    return low, up

def regLabels(k, w, a):
    ''' labels(k, w, a)
            Return k labels on questions for a single worker
    '''
    if a: # Attackers
        r = list(np.random.gamma(.42, .1, len(k)))
        #r = [rnd.uniform(0, 1) for i in range(len(k))]
        low = [w[0] - r[i] for i in range(len(k))]
        up = [w[0] + r[i] for i in range(len(k))]
    else:
        r = [rnd.uniform(0, .05) for i in range(len(k))]
        low = [w[0] - r[i] for i in range(len(k))]
        up = [w[0] + r[i] for i in range(len(k))]
    for i in range(len(k)):
        if low[i] > up[i]:
            low[i], up[i] = up[i], low[i]
        if low[i] < 0:
            low[i] = 0
        if up[i] > 1:
            up[i] = 1
            
    return low, up

def dist(a, b):
    ''' Distance formula for intervals '''
    return abs((a[0]+a[1])/2 - (b[0]+b[1])/2) + abs((a[1]-a[0])/2 - (b[1]-b[0])/2)

def main(k=100):
    # Import dataset
    data = pd.read_csv("Income94.csv")
    
    # Split dataset into gold and regular questions
    q = data["Class"]
    goldQ = q.sample(frac = 1/3)
    regQ = q.drop(goldQ.index)
    
    # Generate 100 workers with various reliabilities
    # workers = []
    # seeds = [rnd.uniform(0, 1) for i in range(k)]
    # for i in range(k):
    #     low, up = ivls(i, seeds[i], )
    #     w = Worker(low, up)
    #     workers.append([w.center, w.confidence, w.stability, w.entropy])
        
    # # Save workers to dataset
    # df = pd.DataFrame(workers, columns=['Correctness', 'Confidence', 'Stability', 'Predictability'])
    # df.to_excel('workers.xlsx')
    
    # Read workers from dataset
    workers = pd.read_excel("workers.xlsx")
    workers = workers.iloc[:,1:]

    # Sample attackers from workers
    ans = workers.sample(20).index
    workers = workers.values.tolist()
    k = len(workers)
    
    # Run workers on gold questions
    goldWorkers = []
    # golds = rnd.sample(list(goldQ), 200)
    for i in range(k):
        gLow, gUp = goldLabels(list(goldQ), workers[i])
        w = Worker(gLow, gUp)
        goldWorkers.append([w.mean, w.var])
    
    # Run workers on regular questions
    regWorkers = []
    a = False
    for i in range(k):
        if i in ans:
            a = True
        else:
            a = False
        rLow, rUp = regLabels(list(regQ), workers[i], a)
        w = Worker(rLow, rUp)
        regWorkers.append([w.mean, w.var])
    
    '''df = pd.DataFrame(goldWorkers, columns=['Gold Mean', 'Gold Var'])
    dfr = pd.DataFrame(regWorkers, columns=['Reg Mean', 'Reg Var'])
    results = pd.concat([df, dfr], axis=1)
    results.to_excel('anomalyIncome94.xlsx')'''
    
    # Run t-test and report anomalys
    anomX, normX, anomY, normY = [], [], [], []
    tEyes, eyes = [], []
    for i in range(k):
        # Test Statistic
        T = dist(goldWorkers[i][0], regWorkers[i][0])/ \
            (math.sqrt(goldWorkers[i][1]/len(goldQ) + regWorkers[i][1]/len(regQ)))
        
        # Degrees of Freedom
        dof = ((goldWorkers[i][1]/len(goldQ) + regWorkers[i][1]/len(regQ)) *                \
            (goldWorkers[i][1]/len(goldQ) + regWorkers[i][1]/len(regQ))) /                  \
            ((goldWorkers[i][1]/len(goldQ)) * (goldWorkers[i][1]/len(goldQ)) /              \
            (len(goldQ)-1) + (regWorkers[i][1]/len(regQ)) * (regWorkers[i][1]/len(regQ)) /  \
            (len(regQ)-1))
       
        # Critical Value
        tCrit = t.ppf(q=1-.05/2,df=dof)
        if T > tCrit:
            tEyes.append(i)
    
    # Run F-test and report anomaly
    for i in range(k):
        # Test Statistic
        F = regWorkers[i][1] / goldWorkers[i][1]
        fCrit = f.ppf(q=1-.05/2,dfn=len(regQ)-1, dfd=len(goldQ)-1)
        
        if F > fCrit:
            if i in tEyes:
                eyes.append(i)
                anomX.append(workers[i][0])
                anomY.append(workers[i][1])
        normX.append(workers[i][0])
        normY.append(workers[i][1])
          
    # Plot the workers and attackers
    plt.scatter(normX, normY, color='#2CBDFE')
    plt.scatter(anomX, anomY, color='#661D98', marker='x', s=70)
    plt.title("Income94", fontsize=15)
    plt.xlabel("Correctness", fontsize=15)
    plt.ylabel("Confidence", fontsize=15)
    plt.show()
    
    ans = list(ans)
    ans.sort()
    print(f"Generated Anomalies:\n{ans}\n")
    print(f"Found Anomalies:\n{eyes}")

if __name__ == '__main__':
    main()