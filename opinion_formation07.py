#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 16:29:52 2020

@author: ckadelka
"""

import numpy as np
import networkx as nx
import random
import math
import itertools
import scipy.optimize as opti
import matplotlib.pyplot as plt

def opinion_clustering(G,c=0.5,s=0,nsim=0,o=[]):
    '''
    G = nx.graph = network
    c = distribution of binary attribute, in (0,1)
    s = strength of opinion formation, in [0,1]
    '''
    N = len(G.nodes)
    nodes = list(range(N))
    #edges = list(G.edges)
    
    if nsim==0:
        nsim=N #as in Salathe, Bonhoeffer 2008
    
    #assigment of opinion based on parameter c
    if o==[]:
        opinions = np.random.random(N)>c
    else:
        opinions = np.array(o).copy()
    
    assert sum(opinions) not in [0,N],'Error: all nodes share the same opinion'
    
    #opinion formation
    ds = [sum([opinions[node]!=opinions[neighbor] for neighbor in G.neighbors(node)])/G.degree(node) for node in nodes]
    
    node = int(random.random()*N)
    opinion_node = opinions[node]
    nr_swaps = 0
    for i in range(nsim):
        while opinion_node!=opinions[node]:
            node = int(random.random()*N)
        if random.random()<ds[node]*s:
            opinions[node] = ~opinions[node]
            opinion_node = opinions[node]
            ds[node] = 1-ds[node]
            nr_swaps+=1
            for neighbor in G.neighbors(node):
                size = G.degree[neighbor]
                if opinion_node==opinions[neighbor]:
                    ds[neighbor] = (ds[neighbor]*size-1)/size
                else:
                    ds[neighbor] = (ds[neighbor]*size+1)/size
                #ds[neighbor] = sum([opinions[neighbor]!=opinions[neighbor2] for neighbor2 in G.neighbors(neighbor)])/G.degree(neighbor)
        node = int(random.random()*N)
    return opinions,nr_swaps

def opinion_clustering_extended(G,c=0.5,s=0,nsim=0,o=[],threshold=1):
    '''
    G = nx.graph = network
    c = distribution of binary attribute, in (0,1)
    s = strength of opinion formation, in [0,1]
    '''
    N = len(G.nodes)
    nodes = list(range(N))
    #edges = list(G.edges)
    n_edges = G.size()
    
    if nsim==0:
        nsim=N #as in Salathe, Bonhoeffer 2008
    
    #assigment of opinion based on parameter c
    if o==[]:
        opinions = np.random.random(N)>c
    else:
        opinions = np.array(o).copy()
        
    
    assert sum(opinions) not in [0,N],'Error: all nodes share the same opinion'
    
    #opinion formation
    ds = [sum([opinions[node]!=opinions[neighbor] for neighbor in G.neighbors(node)])/G.degree(node) for node in nodes]
    degrees = [G.degree(node) for node in nodes]
    count_same_edges = n_edges-np.dot(degrees,ds)/2
    
    node = int(random.random()*N)
    opinion_node = opinions[node]
    nr_swaps = 0
    for i in range(nsim):
        while opinion_node!=opinions[node]:
            node = int(random.random()*N)
        if random.random()<ds[node]*s:
            opinions[node] = ~opinions[node]
            opinion_node = opinions[node]
            count_same_edges += (2*ds[node]-1)*degrees[node]
            ds[node] = 1-ds[node]
            nr_swaps+=1
            for neighbor in G.neighbors(node):
                size = G.degree[neighbor]
                if opinion_node==opinions[neighbor]:
                    ds[neighbor] = (ds[neighbor]*size-1)/size
                else:
                    ds[neighbor] = (ds[neighbor]*size+1)/size
            
        node = int(random.random()*N)
        if count_same_edges/n_edges>=threshold: #stopping criterion to stop at a certain proportion of edges with same opinion
            break
    return opinions,nr_swaps
            
def get_proportion_of_edges_with_same_opinion(G,opinions):
    count_same = 0
    for (a,b) in G.edges:
        if opinions[a]==opinions[b]:
            count_same+=1
    return count_same/G.size()

def draw_graph_with_opinions(G,opinions):
    pos = nx.drawing.layout.kamada_kawai_layout(G)
    f,ax = plt.subplots()
    nx.draw_networkx(G,pos=pos,with_labels=False,ax=ax,node_size=0)
    #ax.legend(loc='best',title='contacts')
    ax.axis('off')
    markersize = 7
    pos_array = np.array([pos[i] for i in range(N)])
    which = opinions
    line3, = ax.plot(pos_array[which,0],pos_array[which,1],'ks',markersize=markersize)
    line4, = ax.plot(pos_array[~which,0],pos_array[~which,1],'ro',markersize=markersize)
    return f,ax




#now 2D
def get_correlated_binomial_rvs(c1,c2,corr,N=1):
    '''stats.stackexchange.com/questions/284996/generating-correlated-binomial-random-variables'''
    P00 = (1-c1)*(1-c2) + corr*math.sqrt(c1*c2*(1-c1)*(1-c2));
    P01 = 1 - c1 - P00;
    P10 = 1 - c2 - P00;
    P11 = P00 + c1 + c2 - 1;
    PROBS = [P00, P01, P10, P11]
    assert min(PROBS) >= 0,'Error: corr is not in the allowable range'
    
    #Generate the output
    RAND = random.choices(range(4),PROBS,k=N)
    rv1 = [el//2 for el in RAND]
    rv2 = [el%2 for el in RAND]
    return rv1,rv2

def get_correlated_binomial_pdf_3d_scipy(p1,p2,p3,c12=0,c13=0,c23=0,c123=0):
    def equations(p,*args):
        p000, p001, p010, p011, p100, p101, p110, p111 = p
        p1,p2,p3,c12,c13,c23,c123 = args[0] 
        return (p000 + p001 + p010 + p011 + p100 + p101 + p110 + p111 - 1, 
                p100 + p101 + p110 + p111 - p1, 
                p010 + p011 + p110 + p111 - p2, 
                p001 + p011 + p101 + p111 - p3, 
                (p111 + p110 - p1*p2)/math.sqrt(p1*p2*(1-p1)*(1-p2)) - c12, 
                (p111 + p101 - p1*p3)/math.sqrt(p1*p3*(1-p1)*(1-p3)) - c13, 
                (p111 + p011 - p2*p3)/math.sqrt(p2*p3*(1-p2)*(1-p3)) - c23, 
                ((1-p1-p2-p3)*p111 - p1*p011 - p2*p101 - p3*p110 + 2*p1*p2*p3)/math.sqrt(p1*p2*p3*(1-p1)*(1-p2)*(1-p3)) - c123)
    matrix = opti.fsolve(equations,[1/8]*8,args=[p1,p2,p3,c12,c13,c23,c123])
    return list(matrix)

def get_correlated_binomial_rvs_3d(p1,p2,p3,corr12,corr13,corr23,corr123=0,N=1):
    '''stats.stackexchange.com/questions/284996/generating-correlated-binomial-random-variables'''
    
    PROBS = get_correlated_binomial_pdf_3d_scipy(p1,p2,p3,corr12,corr13,corr23,corr123)
    assert min(PROBS) >= 0,'Error: corr is not in the allowable range'
    
    #Generate the output
    RAND = random.choices(range(8),PROBS,k=N)
    rv1 = [el//4 for el in RAND]
    rv2 = [(el%4)//2 for el in RAND]
    rv3 = [el%2 for el in RAND]
    return rv1,rv2,rv3

def get_allowable_range(c1,c2,depth=20):
    corr=0.5
    for i in range(2,2+depth):
        P00 = (1-c1)*(1-c2) + corr*math.sqrt(c1*c2*(1-c1)*(1-c2))
        P01 = 1 - c1 - P00
        P10 = 1 - c2 - P00
        P11 = P00 + c1 + c2 - 1
        if min([P00, P01, P10, P11])>0:
            corr+=1/2**i
        else:
            corr-=1/2**i
    if min([P00, P01, P10, P11])<0:
        corr-=1/2**i
    corr2=-0.5
    for i in range(2,2+depth):
        P00 = (1-c1)*(1-c2) + corr2*math.sqrt(c1*c2*(1-c1)*(1-c2))
        P01 = 1 - c1 - P00
        P10 = 1 - c2 - P00
        P11 = P00 + c1 + c2 - 1
        if min([P00, P01, P10, P11])>0:
            corr2-=1/2**i
        else:
            corr2+=1/2**i
    if min([P00, P01, P10, P11])<0:
        corr2+=1/2**i
    return corr2,corr

def get_expected_correlation(c1,c2,n_sample=1000):
    RV1 = np.ones(n_sample)
    RV2 = np.ones(n_sample)
    RV1[np.random.choice(range(n_sample),int(round(c1*n_sample)),False)] = 0
    RV2[np.random.choice(range(n_sample),int(round(c2*n_sample)),False)] = 0
    return np.corrcoef(RV1,RV2)[0,1]

def plot_allowable_range_given_fixed_c1(c1):
    xs = np.linspace(0.001,0.999,1000)
    ys = np.array([get_allowable_range(c1,el) for el in xs]).T
    f,ax = plt.subplots()
    ax.fill_between(xs,ys[0],ys[1])
    ax.set_xlabel('c2')
    ax.set_ylabel('correlation')
    return f,ax

def plot_allowable_range_given_fixed_c1s(c1s=[0.5,0.7,0.9,0.95]):
    colors= ['b','r','g','k']
    f,ax = plt.subplots()
    for i,c1 in enumerate(c1s):
        xs = np.linspace(0.0001,0.9999,1000)
        ys = np.array([get_allowable_range(c1,el) for el in xs]).T
        ax.plot(xs,ys[0],'-',color=colors[i])
        ax.plot(xs,ys[1],'-',color=colors[i])  
    ax.set_xlabel('c2')
    ax.set_ylabel('correlation')
    return f,ax

def plot_allowable_range_given_fixed_c1s_nice(c1s=[0.5,0.7,0.9,0.99]):
    colors= ['b','r','g','k']
    lss = ['-','-.','--',':']
    f,ax = plt.subplots()
    for i,c1 in enumerate(c1s):
        xs = np.linspace(0.0001,0.9999,1000)
        ys = np.array([get_allowable_range(c1,el) for el in xs]).T
        ax.plot(xs,ys[0],'-',color=colors[i],ls=lss[i],label=str(c1))
        ax.plot(xs,ys[1],'-',color=colors[i],ls=lss[i],label='__nolabel__')  
    ax.set_xlabel(r'$p_2 = P(X_2=1)$')
    ax.set_ylabel(r'correlation $\sigma_{12}$')
    ax.legend(loc='center',frameon=False,title=r'$p_1= P(X_1=1)$',ncol=4,bbox_to_anchor=(0.5,1.1))
    plt.savefig('allowable_range.pdf',bbox_inches = "tight") 
    return f,ax

def opinion_clustering_2d_v2b(G,cs=[0.5,0.5],corr=0,nsim=0,o=[],thresholds=[1,1],RETURN_ARGMAX=True,exponent=1,accuracy=0.01):
    '''
    G = nx.graph = network
    cs = distribution of binary opinion, for each opinion type in (0,1)
    corr = correlation between opinion types (if d>2, correlation matrix, if d==2, correlation coefficient)
    nsim = stopping criterion: number of opinion swaps considered 
    o = (optional) 
    thresholds = optional stopping criterion: desired proportion of edges with same opinion, for each opinion type in [0,1]   
    
    '''
    N = len(G.nodes)
    nodes = list(range(N))
    #edges = list(G.edges)
    n_edges = G.size()
    d = 2
    
    if nsim==0:
        nsim=N #as in Salathe, Bonhoeffer 2008

    #assigment of opinion based on parameters in cs
    c1,c2 = cs
    range1,range2 = get_allowable_range(c1,c2,10)
    assert range1<=corr and corr<=range2,'Error: chosen correlation structure is not possible given the choices of cs'

    if o==[]:
        opinions1,opinions2 = get_correlated_binomial_rvs(c1,c2,corr,N=N)
        opinions = np.array([opinions1,opinions2],dtype=bool)
    else:
        opinions = np.array(o,dtype=bool)
        for i,opinion in enumerate(opinions):
            assert sum(opinion) not in [0,N],'Error: all nodes share the same opinion w.r.t attribute %i' % (i+1)

    while sum([sum(opinion) in [0,N] for opinion in opinions])>0:
        opinions1,opinions2 = get_correlated_binomial_rvs(c1,c2,corr,N=N)
        opinions = np.array([opinions1,opinions2],dtype=bool)        

    which00 = list(np.arange(N)[np.bitwise_and(opinions[0]==0,opinions[1]==0)])
    which01 = list(np.arange(N)[np.bitwise_and(opinions[0]==0,opinions[1]==1)])
    which10 = list(np.arange(N)[np.bitwise_and(opinions[0]==1,opinions[1]==0)])
    which11 = list(np.arange(N)[np.bitwise_and(opinions[0]==1,opinions[1]==1)])
    prop_which00 = len(which00)/N
    prop_which01 = len(which01)/N
    prop_which10 = len(which10)/N
    prop_which0x = (prop_which00+prop_which01)
    prop_whichx0 = (prop_which00+prop_which10)

    #opinion formation
    degrees = [G.degree(node) for node in nodes]
    ds = [[sum([opinions[index][node]!=opinions[index][neighbor] for neighbor in G.neighbors(node)])/G.degree(node) for node in nodes] for index in range(d)]
    which_list = [which00,which01,which10,which11]
    ds_list = [[[ds[index][node] for node in which] for which in which_list] for index in range(d)]

    
    count_same_edges = [n_edges-np.dot(degrees,ds[index])/2 for index in range(d)]

    dict_pos = dict()
    for vector in which_list:
        for i,el in enumerate(vector):
            dict_pos.update({el:i})

    dict_where = dict()
    for i,vector in enumerate(which_list):
        for el in vector:
            dict_where.update({el:i})

    #argmax_which = [which00.copy(),which01.copy(),which10.copy(),which11.copy()]
    argmax_opinions = opinions.copy()
    min_difference = 1e10
    #print(max_score)
        
    opinions_being_optimized = [1,0]
    opinions_still_being_optimized = opinions_being_optimized[:]
    nr_swaps = 0
    index = opinions_still_being_optimized[0]
    #matrix = np.array(pd.crosstab(np.array(opinions[0],dtype=bool),np.array(opinions[1],dtype=bool)))
    increase_clustering_for_this_index = [count_same_edges[index]/n_edges<thresholds[index] for index in range(d)]

    while opinions_still_being_optimized!=[]:
    
        if index==0:
            rv = random.random()
            if rv<prop_whichx0:
                whereA,whereB = 0,2
            else:
                whereA,whereB = 1,3
        elif index==1:
            rv = random.random()
            if rv<prop_which0x:
                whereA,whereB = 0,1
            else:
                whereA,whereB = 2,3

        whichA,whichB = which_list[whereA],which_list[whereB]
        dsA,dsB = ds_list[index][whereA],ds_list[index][whereB]
                
        if increase_clustering_for_this_index[index]:
            nodeA = random.choices(whichA,weights=list(map(lambda x: x**exponent,dsA)))[0]
            nodeB = random.choices(whichB,weights=list(map(lambda x: x**exponent,dsB)))[0]  
        else:
            nodeA = random.choices(whichA,weights=list(map(lambda x: (1-x)**exponent,dsA)))[0]
            nodeB = random.choices(whichB,weights=list(map(lambda x: (1-x)**exponent,dsB)))[0]              
        
        for node in [nodeA,nodeB]:
            opinions[index][node] = not opinions[index][node]
            count_same_edges[index] += (2*ds_list[index][dict_where[node]][dict_pos[node]]-1)*degrees[node]
            nr_swaps+=1
            ds_list[index][dict_where[node]][dict_pos[node]] = 1-ds_list[index][dict_where[node]][dict_pos[node]]
            for neighbor in G.neighbors(node):
                whereNeighbor = dict_where[neighbor]
                posNeighbor = dict_pos[neighbor]
                if opinions[index][node]==opinions[index][neighbor]:
                    ds_list[index][whereNeighbor][posNeighbor] = (ds_list[index][whereNeighbor][posNeighbor]*degrees[neighbor]-1)/degrees[neighbor]
                else:
                    ds_list[index][whereNeighbor][posNeighbor] = (ds_list[index][whereNeighbor][posNeighbor]*degrees[neighbor]+1)/degrees[neighbor]
                    
        whichA[dict_pos[nodeA]] = nodeB
        whichB[dict_pos[nodeB]] = nodeA
        for index_index in opinions_being_optimized:
            ds_list[index_index][dict_where[nodeA]][dict_pos[nodeA]],ds_list[index_index][dict_where[nodeB]][dict_pos[nodeB]] =  ds_list[index_index][dict_where[nodeB]][dict_pos[nodeB]],ds_list[index_index][dict_where[nodeA]][dict_pos[nodeA]]
        dict_pos[nodeA],dict_pos[nodeB] = dict_pos[nodeB],dict_pos[nodeA] #swap
        dict_where[nodeA],dict_where[nodeB] = dict_where[nodeB],dict_where[nodeA] #swap

        #assert np.all(np.array(pd.crosstab(np.array(opinions[0],dtype=bool),np.array(opinions[1],dtype=bool))) == matrix)


        dummy = sum([abs(count_same_edges[index]/n_edges-thresholds[index]) for index in opinions_being_optimized])
        if dummy < min_difference:
            min_difference = dummy
            argmax_opinions = opinions.copy()
            #print(nr_swaps,min_difference)
            if min_difference<accuracy:
                break

        if count_same_edges[index]/n_edges>=thresholds[index]: #stopping criterion to stop at a certain proportion of edges with same opinion
            increase_clustering_for_this_index[index]=False
            opinions_still_being_optimized.remove(index)
            if opinions_still_being_optimized==[]:
                break
        else:
            increase_clustering_for_this_index[index]=True
        index = random.choice(opinions_still_being_optimized)
        nr_swaps+=1
        if nr_swaps>=nsim:
            break


    #print(nr_swaps,np.array(count_same_edges)/n_edges)
    return argmax_opinions if RETURN_ARGMAX else opinions,nr_swaps


def opinion_clustering_2d_v3(G,cs=[0.5,0.5],corr=0,nsim=0,o=[],thresholds=[1,1],RETURN_ARGMAX=True,exponent=1,accuracy=0.01):
    '''
    G = nx.graph = network
    cs = distribution of binary opinion, for each opinion type in (0,1)
    corr = correlation between opinion types (if d>2, correlation matrix, if d==2, correlation coefficient)
    nsim = stopping criterion: number of opinion swaps considered 
    o = (optional) 
    thresholds = optional stopping criterion: desired proportion of edges with same opinion, for each opinion type in [0,1]   
    
    '''
    N = len(G.nodes)
    nodes = list(range(N))
    #edges = list(G.edges)
    n_edges = G.size()
    d = 2
    
    if nsim==0:
        nsim=N #as in Salathe, Bonhoeffer 2008

    #assigment of opinion based on parameters in cs
    c1,c2 = cs
    range1,range2 = get_allowable_range(c1,c2,10)
    assert range1<=corr and corr<=range2,'Error: chosen correlation structure is not possible given the choices of cs'

    if o==[]:
        opinions1,opinions2 = get_correlated_binomial_rvs(c1,c2,corr,N=N)
        opinions = np.array([opinions1,opinions2],dtype=bool)
    else:
        opinions = np.array(o,dtype=bool)
        for i,opinion in enumerate(opinions):
            assert sum(opinion) not in [0,N],'Error: all nodes share the same opinion w.r.t attribute %i' % (i+1)

    while sum([sum(opinion) in [0,N] for opinion in opinions])>0:
        opinions1,opinions2 = get_correlated_binomial_rvs(c1,c2,corr,N=N)
        opinions = np.array([opinions1,opinions2],dtype=bool)        

    which00 = list(np.arange(N)[np.bitwise_and(opinions[0]==0,opinions[1]==0)])
    which01 = list(np.arange(N)[np.bitwise_and(opinions[0]==0,opinions[1]==1)])
    which10 = list(np.arange(N)[np.bitwise_and(opinions[0]==1,opinions[1]==0)])
    which11 = list(np.arange(N)[np.bitwise_and(opinions[0]==1,opinions[1]==1)])

    #opinion formation
    degrees = [G.degree(node) for node in nodes]
    ds = [[sum([opinions[index][node]!=opinions[index][neighbor] for neighbor in G.neighbors(node)])/G.degree(node) for node in nodes] for index in range(d)]
    which_list = [which00,which01,which10,which11]
    ds_list = [[[ds[index][node] for node in which] for which in which_list] for index in range(d)]
    len_which_list = list(map(len,which_list))
    
    count_same_edges = [n_edges-np.dot(degrees,ds[index])/2 for index in range(d)]

    dict_pos = dict()
    for vector in which_list:
        for i,el in enumerate(vector):
            dict_pos.update({el:i})

    dict_where = dict()
    for i,vector in enumerate(which_list):
        for el in vector:
            dict_where.update({el:i})

    #argmax_which = [which00.copy(),which01.copy(),which10.copy(),which11.copy()]
    argmax_opinions = opinions.copy()
    min_difference = 1e10
    #print(max_score)
        
    opinions_being_optimized = list(range(d))
    opinions_still_being_optimized = opinions_being_optimized[:]
    nr_swaps = 0
    index = opinions_still_being_optimized[0]
    #print('\n\naaa')
    increase_clustering_for_this_index = [count_same_edges[index]/n_edges<thresholds[index] for index in range(d)]
    while opinions_still_being_optimized!=[]:
        
        whereA = random.choices(range(2**d),len_which_list)[0]
        whereB = whereA
        while whereB==whereA:
            whereB = random.choices(range(2**d),len_which_list)[0]
        whichA,whichB = which_list[whereA],which_list[whereB]
        
        indices_changed_this_turn = []
        if whereA//2 != whereB//2:
            indices_changed_this_turn.append(0)
        if whereA%2 != whereB%2:
            indices_changed_this_turn.append(1)            
                
        weightsA = [sum([ds_list[index][whereA][i]**exponent if increase_clustering_for_this_index[indices_changed_this_turn[0]] else (1 - ds_list[indices_changed_this_turn[0]][whereA][i])**exponent for index in indices_changed_this_turn]) for i in range(len(whichA))]
        weightsB = [sum([ds_list[index][whereB][i]**exponent if increase_clustering_for_this_index[indices_changed_this_turn[0]] else (1 - ds_list[indices_changed_this_turn[0]][whereB][i])**exponent for index in indices_changed_this_turn]) for i in range(len(whichB))]

        nodeA = random.choices(whichA,weights=weightsA)[0]
        nodeB = random.choices(whichB,weights=weightsB)[0]  
        
        for index in indices_changed_this_turn:
            for node in [nodeA,nodeB]:
                opinions[index][node] = not opinions[index][node]
                count_same_edges[index] += (2*ds_list[index][dict_where[node]][dict_pos[node]]-1)*degrees[node]
                ds_list[index][dict_where[node]][dict_pos[node]] = 1-ds_list[index][dict_where[node]][dict_pos[node]]
                for neighbor in G.neighbors(node):
                    whereNeighbor = dict_where[neighbor]
                    posNeighbor = dict_pos[neighbor]
                    if opinions[index][node]==opinions[index][neighbor]:
                        ds_list[index][whereNeighbor][posNeighbor] = (ds_list[index][whereNeighbor][posNeighbor]*degrees[neighbor]-1)/degrees[neighbor]
                    else:
                        ds_list[index][whereNeighbor][posNeighbor] = (ds_list[index][whereNeighbor][posNeighbor]*degrees[neighbor]+1)/degrees[neighbor]

                #assert np.max(list(map(max,ds_list[index])))<=1.00001

                        
        whichA[dict_pos[nodeA]],whichB[dict_pos[nodeB]] = nodeB,nodeA
        for index_index in opinions_being_optimized:
            ds_list[index_index][dict_where[nodeA]][dict_pos[nodeA]],ds_list[index_index][dict_where[nodeB]][dict_pos[nodeB]] =  ds_list[index_index][dict_where[nodeB]][dict_pos[nodeB]],ds_list[index_index][dict_where[nodeA]][dict_pos[nodeA]]
        dict_pos[nodeA],dict_pos[nodeB] = dict_pos[nodeB],dict_pos[nodeA] #swap
        dict_where[nodeA],dict_where[nodeB] = dict_where[nodeB],dict_where[nodeA] #swap
        
        dummy = sum([abs(count_same_edges[index]/n_edges-thresholds[index]) for index in opinions_being_optimized])
        if dummy < min_difference:
            min_difference = dummy
            argmax_opinions = opinions.copy()
            #print(nr_swaps,min_difference)
            if min_difference<accuracy:
                break

        for index in indices_changed_this_turn:
            if count_same_edges[index]/n_edges>=thresholds[index]: #stopping criterion to stop at a certain proportion of edges with same opinion
                try:
                    opinions_still_being_optimized.remove(index)
                except ValueError: # if the index has already been removed
                    pass
                increase_clustering_for_this_index[index]=False
            else:
                increase_clustering_for_this_index[index]=True
        nr_swaps+=1
        if nr_swaps>=nsim:
            break
    #print(np.array(pd.crosstab(np.array(opinions[0],dtype=bool),np.array(opinions[1],dtype=bool))))
    #print(thresholds,get_proportion_of_edges_with_same_opinion_2d(G,opinions))
    #print(thresholds,get_proportion_of_edges_with_same_opinion_2d(G,argmax_opinions))
    #print('aaaa\n\n')
    #print(nr_swaps)
    #print(nr_swaps,np.array(count_same_edges)/n_edges)
    
    return argmax_opinions if RETURN_ARGMAX else opinions,nr_swaps

def opinion_clustering_3d_v2b(G,ps=[0.5,0.5,0.5],c12=0,c13=0,c23=0,c123=0,nsim=0,o=[],thresholds=[1,1,1],RETURN_ARGMAX=True,exponent=1,accuracy=0.01):
    '''
    G = nx.graph = network
    cs = distribution of binary opinion, for each opinion type in (0,1)
    corr = correlation between opinion types (if d>2, correlation matrix, if d==2, correlation coefficient)
    nsim = stopping criterion: number of opinion swaps considered 
    o = (optional) 
    thresholds = optional stopping criterion: desired proportion of edges with same opinion, for each opinion type in [0,1]   
    
    '''
    N = len(G.nodes)
    nodes = list(range(N))
    #edges = list(G.edges)
    n_edges = G.size()
    d = 3
    
    if nsim==0:
        nsim=N #as in Salathe, Bonhoeffer 2008

    #assigment of opinion based on parameters in cs
    p1,p2,p3 = ps
    PROBS = get_correlated_binomial_pdf_3d_scipy(p1,p2,p3,c12,c13,c23,c123)
    assert min(PROBS)>=0,'Error: chosen correlation structure is not possible given the choices of cs'

    if o==[]:
        opinions1,opinions2,opinions3 = get_correlated_binomial_rvs_3d(p1,p2,p3,c12,c13,c23,c123,N=N)
        opinions = np.array([opinions1,opinions2,opinions3],dtype=bool)
    else:
        opinions = np.array(o,dtype=bool)
        for i,opinion in enumerate(opinions):
            assert sum(opinion) not in [0,N],'Error: all nodes share the same opinion w.r.t attribute %i' % (i+1)

    while sum([sum(opinion) in [0,N] for opinion in opinions])>0:
        opinions1,opinions2,opinions3 = get_correlated_binomial_rvs_3d(p1,p2,p3,c12,c13,c23,c123,N=N)
        opinions = np.array([opinions1,opinions2,opinions3],dtype=bool)       

    which_decimal = np.dot(2**np.arange(d-1,-1,-1),opinions)
    which_list = [list(np.arange(N)[which_decimal==nr]) for nr in range(2**d)]
    prop_which_actual = np.array(list(map(len,which_list)))
    T = np.array(list(itertools.product([0, 1], repeat=d)))    
    which_zero_per_index = [[np.arange(2**d)[T[:,index]==0],np.arange(2**d)[T[:,index]==1]] for index in range(d)]
    prop_which_actual_fixed_index = [prop_which_actual[np.arange(2**d)[T[:,index]==0]] + prop_which_actual[np.arange(2**d)[T[:,index]==1]] for index in range(d)]
    
    #opinion formation
    degrees = [G.degree(node) for node in nodes]
    ds = [[sum([opinions[index][node]!=opinions[index][neighbor] for neighbor in G.neighbors(node)])/G.degree(node) for node in nodes] for index in range(d)]
    ds_list = [[[ds[index][node] for node in which] for which in which_list] for index in range(d)]

    
    count_same_edges = [n_edges-np.dot(degrees,ds[index])/2 for index in range(d)]

    dict_pos = dict()
    for vector in which_list:
        for i,el in enumerate(vector):
            dict_pos.update({el:i})

    dict_where = dict()
    for i,vector in enumerate(which_list):
        for el in vector:
            dict_where.update({el:i})

    #argmax_which = [which00.copy(),which01.copy(),which10.copy(),which11.copy()]
    argmax_opinions = opinions.copy()
    min_difference = 1e10
    #print(max_score)

    opinions_being_optimized = [0,1,2]
    opinions_still_being_optimized = opinions_being_optimized[:]
    nr_swaps = 0
    index = opinions_still_being_optimized[0]
    
    
    increase_clustering_for_this_index = [count_same_edges[index]/n_edges<thresholds[index] for index in range(d)]

    while opinions_still_being_optimized!=[]:
        dummy = random.choices(list(range(2**(d-1))),weights=prop_which_actual_fixed_index[index])[0]
        whereA,whereB = which_zero_per_index[index][0][dummy],which_zero_per_index[index][1][dummy]

        whichA,whichB = which_list[whereA],which_list[whereB]
        dsA,dsB = ds_list[index][whereA],ds_list[index][whereB]
                
        nodeA = random.choices(whichA,weights=list(map(lambda x: x**exponent,dsA)))[0]
        nodeB = random.choices(whichB,weights=list(map(lambda x: x**exponent,dsB)))[0]  
                    
        for node in [nodeA,nodeB]:
            opinions[index][node] = not opinions[index][node]
            count_same_edges[index] += (2*ds_list[index][dict_where[node]][dict_pos[node]]-1)*degrees[node]
            ds_list[index][dict_where[node]][dict_pos[node]] = 1-ds_list[index][dict_where[node]][dict_pos[node]]
            for neighbor in G.neighbors(node):
                whereNeighbor = dict_where[neighbor]
                posNeighbor = dict_pos[neighbor]
                if opinions[index][node]==opinions[index][neighbor]:
                    ds_list[index][whereNeighbor][posNeighbor] = (ds_list[index][whereNeighbor][posNeighbor]*degrees[neighbor]-1)/degrees[neighbor]
                else:
                    ds_list[index][whereNeighbor][posNeighbor] = (ds_list[index][whereNeighbor][posNeighbor]*degrees[neighbor]+1)/degrees[neighbor]
                
                #assert ds_list[index][whereNeighbor][posNeighbor]>=-0.00001

        #assert np.min(list(map(min,ds_list[index])))>=-0.00001
        
        
        whichA[dict_pos[nodeA]],whichB[dict_pos[nodeB]] = nodeB,nodeA
        for index_index in opinions_still_being_optimized:
            ds_list[index_index][dict_where[nodeA]][dict_pos[nodeA]],ds_list[index_index][dict_where[nodeB]][dict_pos[nodeB]] =  ds_list[index_index][dict_where[nodeB]][dict_pos[nodeB]],ds_list[index_index][dict_where[nodeA]][dict_pos[nodeA]]
        dict_pos[nodeA],dict_pos[nodeB] = dict_pos[nodeB],dict_pos[nodeA] #swap
        dict_where[nodeA],dict_where[nodeB] = dict_where[nodeB],dict_where[nodeA] #swap
        
        dummy = sum([abs(count_same_edges[index]/n_edges-thresholds[index]) for index in opinions_being_optimized])
        if dummy < min_difference:
            min_difference = dummy
            argmax_opinions = opinions.copy()
            #print(nr_swaps,min_difference)
            if min_difference<accuracy:
                break

        if count_same_edges[index]/n_edges>=thresholds[index]: #stopping criterion to stop at a certain proportion of edges with same opinion
            increase_clustering_for_this_index[index]=False
            opinions_still_being_optimized.remove(index)
            if opinions_still_being_optimized==[]:
                break
        else:
            increase_clustering_for_this_index[index]=True
        index = random.choice(opinions_still_being_optimized)
        nr_swaps+=1
        if nr_swaps>=nsim:
            break
        
        
    #print(nr_swaps,np.array(count_same_edges)/n_edges)
    return argmax_opinions if RETURN_ARGMAX else opinions,nr_swaps

def draw_graph_with_opinions_2d(G,opinions):
    pos = nx.drawing.layout.kamada_kawai_layout(G)
    f,ax = plt.subplots()
    nx.draw_networkx(G,pos=pos,with_labels=False,ax=ax,node_size=0,alpha=0.5)
    #ax.legend(loc='best',title='contacts')
    ax.axis('off')
    markersize = 7
    pos_array = np.array([pos[i] for i in range(N)])
    which1 = np.array(opinions[0])
    which2 = np.array(opinions[1])
    colors = ['k','r']
    markers = ['X','o']
    for i in range(2):
        for j in range(2):
            which = np.bitwise_and(which1==i,which2==j)
            ax.plot(pos_array[which,0],pos_array[which,1],color=colors[i],marker=markers[j],ls='',markersize=markersize)
    return f,ax

def get_proportion_of_edges_with_same_opinion_2d(G,opinions):
    return [get_proportion_of_edges_with_same_opinion(G,vec) for vec in opinions]

       