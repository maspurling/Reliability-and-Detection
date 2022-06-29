# -*- coding: utf-8 -*-
"""
    Impact of including and excluding attackers
    on interval-valued workers
    
    Returns graph of F1-Score versus Number of Attackers
    for including and excluding attackers
    
    @author: Makenzie Spurling
"""
import pandas as pd
import random as rnd
import numpy as np
import math
import statistics as st
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
            self.fScore = 0.0
            
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

def labels(g, w, a, b):
    ''' labels(g, w)
            Return labels on gold question g from various workers
    '''
    nIdx = [i[0] for i in w]
    n = [i[1] for i in w]
    r = [rnd.uniform(0, min(n[i][0], 1-n[i][0])) for i in range(len(n))] 
    low = [n[i][0] - r[i] if g == 1 else 1 - (n[i][0] - r[i]) for i in range(len(n))]
    up = [n[i][0] + r[i] if g == 1 else 1 - (n[i][0] + r[i]) for i in range(len(n))] 
    for i in reversed(range(len(n))):
        if nIdx[i] in a:
            if b:
                if n[i][0] < .5:
                    low[i], up[i], n[i][0] = 1-up[i], 1-low[i], 1-n[i][0]
                else:
                    del n[i]
                    del low[i]
                    del up[i]
                    i -= 1
            else:
                if n[i][0] > .50:
                    low[i], up[i] = 1-up[i], 1-low[i]
        if low[i] < 0:
            low[i] = 0
        if up[i] > 1:
            up[i] = 1
        if low[i] > up[i]:
            low[i], up[i] = up[i], low[i]
            
    return low, up, n

def mv(low, up, work):
    ''' mv(low, up, work)
            Returns results of weighted majority voting
    '''
    wtP, wtN = 0, 0
    
    for i in range(len(up)):
        if low[i] >= 0.5 and up[i] > .5:
            wtP += (work[i][0] * ((low[i]+up[i])/2))
        elif low[i] < 0.5 and up[i] <= .5:
            wtN += (work[i][0] * (1 - ((low[i]+up[i])/2)))
    
    if wtP > wtN:
        return True
    elif wtP < wtN:
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
    
    if i == True and g == 1: # TP
        conMat[0] += 1
    elif i == True and g == 0: # FP
        conMat[1] += 1
    elif i == False:
        if g == 0:
            conMat[2] += 1
        else:
            conMat[3] += 1
    return conMat

def getInf(low, up, work):
    ''' getInf(low, up, workers)
            Returns inferences based on IVL's from MV Cen Wt. 
            and PMP Univ. Wt
    '''        
    # Calculate probability on each segment (Assuming uniform)
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
        wtM = sum(pWt[k:])
    elif y != [] and y[0] >= 0.5:
        wtM = sum(pWt)
    else:
        wtM = 0
    if  k != 0 and y[k] - y[k-1] != 0: # for the segment containing 0.5
        tWt = pWt[k-1]/(y[k]-y[k-1])*(y[k] -0.5)
    else:
        tWt = 0
    wtM = wtM + tWt
    
    return mv(low, up, work), wtM

def main():
    # Import dataset
    data = pd.read_csv("Vote.csv")
    
    # Split dataset into gold and regular questions
    q = data["Class"]
    goldQ = q.sample(frac = 1/3)
    
    # Read workers from dataset
    workers = pd.read_excel("workers.xlsx")
    workers = workers.iloc[:,1:]
    
    # Sample attackers from workers
    wrkAtks = list(workers[workers.Confidence > .60].sample(20).index)
    workers = workers[workers.Confidence > .60].values.tolist()

    avg = 20
    fIn, fEx = [], []

    for p in range(len(wrkAtks)):
        for q in range(avg):
            conMatIn, conMatEx = [], []
            for i in goldQ:
                rdWorks = rnd.sample(list(enumerate(workers)), 15)
                atks = rnd.sample(wrkAtks, p+1)
                lowIn, upIn, rdWorksIn = labels(i, rdWorks, atks, False)
                mvInfIn = mv(lowIn, upIn, rdWorksIn)
                lowEx, upEx, rdWorksEx = labels(i, rdWorks, atks, True)
                mvInfEx = mv(lowEx, upEx, rdWorksEx)
                if len(conMatIn) == 0:
                    conMatIn = calMV(mvInfIn, [0, 0, 0, 0], i)
                else:
                    conMatIn = calMV(mvInfIn, conMatIn, i)
                if len(conMatEx) == 0:
                    conMatEx = calMV(mvInfEx, [0, 0, 0, 0], i)
                else:
                    conMatEx = calMV(mvInfEx, conMatEx, i)
            mIn = Measure(conMatIn)
            mEx = Measure(conMatEx)
            if q == 0:
                fIn.append(mIn.fScore)
                fEx.append(mEx.fScore)
            else:
                fIn[p] += mIn.fScore
                fEx[p] += mEx.fScore
    
    fIn = np.array(fIn)
    fEx = np.array(fEx)
    
    fIn = fIn/avg
    fEx = fEx/avg
    y = range(1, len(wrkAtks)+1)
    
    # Plotting
    fig = plt.figure()
    plt.plot(y, fIn, label = "Attacker Included")
    plt.plot(y, fEx, label = "Attacker Excluded")
    plt.xlabel('Number of Attackers', fontsize=16)
    plt.ylabel('F-Score', fontsize=16)
    plt.title("Vote", fontsize=17)
    plt.legend()
    plt.show()
    # fig.savefig('results.png', transparent=True)

if __name__ == '__main__':
    main()