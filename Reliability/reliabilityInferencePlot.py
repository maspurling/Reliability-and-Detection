# -*- coding: utf-8 -*-
"""
    Graphing Inference Methods: Plotting various inference methods 
    based on worker selection
    
    Includes correction of IVLs for workers with high confidence
    and low correctness
    
    Returns plots for accuracy, recall, precision, and fscore for 
    each inference method over an average
    
    * NOTE * - Running this program may take a little bit of time
    
    @author: Makenzie Spurling
"""

import random as rnd
import pandas as pd
import math
import statistics as st
import numpy as np
import matplotlib.pyplot as plt

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
    
def ivls(a, c=0.5, k=35):
    ''' ivls(c, k)
            Return k random intervals with midpoint near c
    '''
    # Random variation for midpoint
    b = rnd.uniform(0, 1)
    mid = [rnd.uniform(c-b, c+b) for i in range(k)]
    for i in range(k):
        if mid[i] > 1:
            mid[i] = 1
        if mid[i] < 0:
            mid[i] = 0
    rad = [min(mid[i], 1-mid[i]) for i in range(k)]

    if a < 150:        
        r =[rnd.uniform(0, rad[i]) for i in range(k)]
    else:
        r =[rnd.uniform(rad[i], 1) for i in range(k)]

    if a < 60 or a > 140:
        _low = [mid[i] - r[i] for i in range(k)]
        _up = [mid[i] + r[i] for i in range(k)]
    elif a < 75 or a > 125:
        _low = [mid[i] - (b+r[i]) for i in range(k)]
        _up = [mid[i] + r[i] for i in range(k)]
    elif a < 90 or a > 110:
        _low = [mid[i] - r[i] for i in range(k)]
        _up = [mid[i] + (b+r[i]) for i in range(k)]
    elif a > 90 or a < 110:
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
            Returns entropy of interval probability
    '''
    e = 0
    for i in range(len(p)):
        if (p[i] != 0): # If a pdf is 0 ignore it
            e -= (p[i]*math.log(p[i]))
    return e

def splitQ(q, frac):
    ''' splitQ(q, frac)
            Return datasets split into gold questions and
            normal questions based on fraction
    '''
    index = int(len(q) * frac)
    goldQ = q[:index]
    normQ = q[index:]
    return goldQ, normQ

def labels(g, w):
    ''' labels(g, w)
            Return labels on gold question g from various workers
    '''
    k = len(w)
    r = [rnd.uniform(0, min(w[i][0], 1-w[i][0])) for i in range(k)] 
    low = [w[i][0] - r[i] if g == 1 else 1 - (w[i][0] - r[i]) for i in range(k)]
    up = [w[i][0] + r[i] if g == 1 else 1 - (w[i][0] + r[i]) for i in range(k)] 
    for i in range(k):
        if low[i] < 0:
            low[i] = 0
        if up[i] > 1:
            up[i] = 1
        if low[i] > up[i]:
            low[i], up[i] = up[i], low[i]
        if w[i][0] < .1 and w[i][1] > .9:
            low[i], up[i], w[i][0] = 1-up[i], 1-low[i], 1-w[i][0]
    return low, up, w

def mv(low, up, work):
    ''' mv(low, up, work)
            Returns results of majority voting
    '''
    bnP, cenP, wtP = 0, 0 ,0
    bnN, cenN, wtN = 0, 0 ,0
    
    for i in range(len(up)):
        if low[i] >= 0.5 and up[i] > .5:
            bnP += 1
            cenP += ((low[i]+up[i])/2)
            wtP += work[i][0] * ((low[i]+up[i])/2)
        elif low[i] < 0.5 and up[i] <= .5:
            bnN += 1
            cenN += (1 - ((low[i]+up[i])/2))
            wtN += work[i][0] * (1 - ((low[i]+up[i])/2))
            
    return mvInf(bnP, bnN), mvInf(cenP, cenN), mvInf(wtP, wtN)

def mvInf(p, n):
    ''' mvInf(p, n)
            Compares MV positive s & negatives to make inference
    '''
    if p > n:
        return True
    elif p < n:
        return False
    else:
        return "Tie"

def calP(i, conMat, g):
    ''' Updates Confusion Matrices for PMP '''
    
    if i > 0.5 and g == 1:
        conMat[0] +=1
    elif i > 0.5 and g == 0:
        conMat[1] +=1
    elif i < 0.5:
        if g == 0:
            conMat[2] += 1
        else:
            conMat[3] += 1
    return conMat

def calMV(i, conMat, g):
    ''' Updates Confusion Matrices for MV '''
    
    if i == True and g == 1:
        conMat[0] += 1
    elif i == True and g == 0:
        conMat[1] += 1
    elif i == False:
        if g == 0:
            conMat[2] += 1
        else:
            conMat[3] += 1
    return conMat

def getInf(low, up, work):
    ''' getInf(low, up, workers)
            Returns inferences based on IVL's from MV Bin, MV Cen, 
            MV Cen Wt. PMP Univ. and PMP Univ. Wt
    '''        
    # Calculate probability on each segment (Assuming uniform)
    pUni = probability_x_unif(low, up)
    pWt = probability_x_unif(low, up, work)
    
    # Obtain the partition  
    y = low + up
    y.sort()

    # Calculating the probability of + matching
    k = 0
    for i in range(1,len(y)):
        if y[i] > 0.5 and y[i-1]<=0.5:
            k = i
            break
        else:
            continue

    if k!= 0:
        uniM = sum(pUni[k:])
        wtM = sum(pWt[k:])
    elif y != [] and y[0] >= 0.5:
        uniM = sum(pUni)
        wtM = sum(pWt)
    else:
        uniM = 0
        wtM = 0
    if  k != 0 and y[k] - y[k-1] != 0: # for the segment containing 0.5
        tUni = pUni[k-1]/(y[k]-y[k-1])*(y[k] -0.5)
        tWt = pWt[k-1]/(y[k]-y[k-1])*(y[k] -0.5)
    else:
        tUni = 0
        tWt = 0
    uniM = uniM + tUni
    wtM = wtM + tWt
    
    return mv(low, up, work), uniM, wtM

def main(k=200):    
    # Read in dataset
    data = pd.read_csv("Car.csv") # "Income94.csv"
    
    # Split the dataset into gold and normal questions
    q = list(data["Class"])
    goldQ, normQ = splitQ(q, 3/10)
    
    # Get workers
    workers = []
    seeds = [rnd.uniform(0, 1) for i in range(k)]
    for i in range(k):
        low, up = ivls(i, seeds[i], 35)
        w = Worker(low, up)
        workers.append([w.center, w.confidence, w.stability, w.entropy])
    
    # workers = pd.read_excel("workers.xlsx")
    # workers = workers.iloc[:,1:]
    # workers = workers.values.tolist()
    
    avg = 40
    ac, f, p, r = [[],[],[],[],[]],[[],[],[],[],[]], \
                  [[],[],[],[],[]], [[],[],[],[],[]]
    conY = []
    # Get random workers, make inferences & confusion matrices
    for q in range(10, 91, 1):
        conY.append(q/100)
        for n in range(avg):
            work = [v for v in workers if v[1] > q/100]
            conMat = []
            for i in goldQ:
                if (len(work) > 10):
                    rdWork = rnd.sample(work, 10)
                else:
                    rdWork = work
                low, up, rdWork = labels(i, rdWork) # Correct Workers
                mvInf, puInf, pwInf = getInf(low, up, rdWork)
                if len(conMat) == 0:
                    conMat.append(calMV(mvInf[0], [0, 0, 0, 0], i))
                    conMat.append(calMV(mvInf[1], [0, 0, 0, 0], i))
                    conMat.append(calMV(mvInf[2], [0, 0, 0, 0], i))
                    conMat.append(calP(puInf, [0, 0, 0, 0], i))
                    conMat.append(calP(pwInf, [0, 0, 0, 0], i))
                else:
                    conMat[0] = calMV(mvInf[0], conMat[0], i)
                    conMat[1] = calMV(mvInf[1], conMat[1], i)
                    conMat[2] = calMV(mvInf[2], conMat[2], i)
                    conMat[3] = calP(puInf, conMat[3], i)
                    conMat[4] = calP(pwInf, conMat[4], i)
            for i in range(len(conMat)):
                m = Measure(conMat[i])
                if n == 0:
                    ac[i].append(m.accuracy)
                    f[i].append(m.fScore)
                    p[i].append(m.precision)
                    r[i].append(m.recall)
                else:
                    ac[i][q-10] += m.accuracy
                    f[i][q-10] += m.fScore
                    p[i][q-10] += m.precision
                    r[i][q-10] += m.recall
                    
    # quals = pd.DataFrame(list(zip(ac, f, p, f)))
    # quals.to_excel("qualitiesCar.xlsx")
    
    ac = np.array(ac)
    f = np.array(f)
    p = np.array(p)
    r = np.array(r)
    
    ac = ac/avg
    f = f/avg
    p = p/avg 
    r = r/avg

    # Plotting
    plt.plot(conY, ac[0], label = "MV", linestyle='-')
    plt.plot(conY, ac[1], label = "IMV", linestyle='--')
    plt.plot(conY, ac[2], label = "WIMV", linestyle='-.')
    plt.plot(conY, ac[3], label = "PMP", linestyle=':')
    plt.plot(conY, ac[4], label = "WPMP", linestyle='-')
    plt.xlabel('Confidence', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.legend()
    plt.show()

    plt.plot(conY, f[0], label = "MV", linestyle='-')
    plt.plot(conY, f[1], label = "IMV", linestyle='--')
    plt.plot(conY, f[2], label = "WIMV", linestyle='-.')
    plt.plot(conY, f[3], label = "PMP", linestyle=':')
    plt.plot(conY, f[4], label = "WPMP", linestyle='-')
    plt.xlabel('Confidence', fontsize=16)
    plt.ylabel('F-Score', fontsize=16)
    plt.legend()
    plt.show()
    
    plt.plot(conY, p[0], label = "MV", linestyle='-')
    plt.plot(conY, p[1], label = "IMV", linestyle='--')
    plt.plot(conY, p[2], label = "WIMV", linestyle='-.')
    plt.plot(conY, p[3], label = "PMP", linestyle=':')
    plt.plot(conY, p[4], label = "WPMP", linestyle='-')
    plt.xlabel('Confidence', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.legend()
    plt.show()
    
    plt.plot(conY, r[0], label = "MV", linestyle='-')
    plt.plot(conY, r[1], label = "IMV", linestyle='--')
    plt.plot(conY, r[2], label = "WIMV", linestyle='-.')
    plt.plot(conY, r[3], label = "PMP", linestyle=':')
    plt.plot(conY, r[4], label = "WPMP", linestyle='-')
    plt.xlabel('Confidence', fontsize=16)
    plt.ylabel('Recall', fontsize=16)
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    main()