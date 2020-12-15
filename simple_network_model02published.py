#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 18:19:45 2020

@author: ckadelka
"""

version='03'

#built-in modules
import sys
import random
import time

#added modules
import numpy as np
import networkx as nx
#import itertools
import opinion_formation07 as opi



output_folder = 'results/simplemodel%s/' % version
output_folder = ''

now = time.time()

try:
    filename = sys.argv[0]
    SLURM_ID = int(sys.argv[1])
except:
    filename = ''
    SLURM_ID = -1#random.randint(0,100)

if len(sys.argv)>2:
    nsim = int(sys.argv[2])
else:
    nsim = 10
    
## parameters for network generation
if len(sys.argv)>3:
    N = int(sys.argv[3]) #network size
else:
    N = 1000
   
if len(sys.argv)>4:
    k = int(sys.argv[4]) #average number of edges per node, initially nodes are connected in a circle, k/2 neighbors to the left, k/2 to the right
else:
    k = 14

if len(sys.argv)>5:
    p_edgechange = float(sys.argv[5]) #average number of edges per node, initially nodes are connected in a circle, k/2 neighbors to the left, k/2 to the right
else:
    p_edgechange = 0.05
    
#network_generating_function = nx.newman_watts_strogatz_graph

## parameters for overall simulation
OUTPUT = 0#True
T_max = 365 #in days

def average_edge_weight_in_susceptible_network(network,activities_S_VS,vaccinated):
    #c = probability_of_close_contact_with_random_person_in_public    
    total_edge_weight_private = 0
    total_weight_of_edges_where_spread_may_occur = 0
    n_edges = 0
    for nodeA,nodeB in network.edges():
        if not vaccinated[nodeA] and not vaccinated[nodeB]:
            total_weight_of_edges_where_spread_may_occur += activities_S_VS[nodeA] * activities_S_VS[nodeB]
        total_edge_weight_private += activities_S_VS[nodeA] * activities_S_VS[nodeB]
        
        n_edges+=1
    return total_edge_weight_private/n_edges,total_weight_of_edges_where_spread_may_occur/n_edges

def model(N,k,p_edgechange,network_generating_function,T_max,beta,recovery_rate,proportion_vaccinated,vaccine_efficacy,private_activity_S,private_activity_V,activity_reduction_I,activity_distancers,proportion_distancers,clustering_vaccine,clustering_distancing,correlation_opinions,INITIALIZE_EQUALLY=False,seed=None,OUTPUT=False):
    if seed==None:
        seed = np.random.randint(1,2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
            
    #build local interaction network and create a list of lists, called neighbors
    network = network_generating_function(N,k,p_edgechange)
    #to draw network: nx.draw(network)
    neighbors = [[] for _ in range(N)]
    for (a,b) in network.edges():
        neighbors[a].append(b)
        neighbors[b].append(a)

    cs = [proportion_vaccinated,proportion_distancers]
    thresholds = [clustering_vaccine,clustering_distancing]
    
    opinions,nr_swaps = opi.opinion_clustering_2d_v2b(network,cs=cs,corr=correlation_opinions,nsim=2*N,thresholds=thresholds,exponent=16,accuracy=0.001)
    actual_clustering_vacc,actual_clustering_dist = opi.get_proportion_of_edges_with_same_opinion_2d(network,opinions)
    vaccinated = list(opinions[0])
    distancing = list(opinions[1])
    actual_correlation = np.corrcoef(vaccinated,distancing)[0,1]
    
    OUTBREAK = False

    S = []
    VS = []#vaccinated but susceptible
    VI = []#vaccinated and immune   
    nextstage = ['I' for _ in range(N)]
    for el in range(N):
        if vaccinated[el]:
            if random.random()<vaccine_efficacy:
                VI.append(el)
                nextstage[el]='S' #eventually, maybe in future version of model immunity wanes... need this line already now to skip these neighbors when deciding who gets infected
            else:
                VS.append(el)
        else:
            S.append(el)
    
    activities_S_VS = [el1*el2 for el1,el2 in zip([activity_distancers if distancing[i] else 1 for i in range(N)],[private_activity_V if vaccinated[i] else private_activity_S for i in range(N)])]
    if INITIALIZE_EQUALLY:
        initial_exposed = random.choice(S+VS)
    else:
        initial_exposed = random.choices(S+VS,weights=[activities_S_VS[el] for el in S+VS])[0]
    
    
    I = [initial_exposed]
    R = []
    try:
        S.remove(initial_exposed)
    except ValueError:
        VS.remove(initial_exposed)
    nextstage[initial_exposed] = 'R'
    
    
    tew,tew_spread = average_edge_weight_in_susceptible_network(network,activities_S_VS,vaccinated)
                
    counter_secondary_infections = [0 for i in range(N)]
    time_infected = [np.nan for i in range(N)]
    time_infected[initial_exposed] = 0
    disease_generation_time = [np.nan for i in range(N)]

    len_VI,len_VS,len_S,len_I,len_R = len(VI),len(VS),len(S),1,0

    if OUTPUT:
        res = [len_VI,len_VS,len_S,len_I,len_R]
        
    
    for t in range(1,T_max+1):
        #go through all symptomatic and check whether they cause new infections among their private or random public contacts
        dict_newly_exposed = {}
        for contagious in I:
            activity_contagious = (1-activity_reduction_I)*activities_S_VS[contagious]
            for neighbor in neighbors[contagious]:
                if nextstage[neighbor]=='I':
                    probability_of_infection = beta * activity_contagious * activities_S_VS[neighbor]
                    if probability_of_infection>random.random():
                        try:
                            dict_newly_exposed[neighbor].append(contagious)
                        except KeyError:
                            dict_newly_exposed.update({neighbor:[contagious]})                

        #disease generation time and who caused infections
        for newly_exposed in dict_newly_exposed: 
            len_dict_newly_exposed = len(dict_newly_exposed[newly_exposed])
            for contagious in dict_newly_exposed[newly_exposed]:
                counter_secondary_infections[contagious] += 1/len_dict_newly_exposed

            if len_dict_newly_exposed>1:
                dummy = []
                for el in dict_newly_exposed[newly_exposed]:
                    dummy.append(t-time_infected[el])
                disease_generation_time[newly_exposed] = np.mean(dummy)
            else:
                disease_generation_time[newly_exposed] = t-time_infected[dict_newly_exposed[newly_exposed][0]]
                time_infected[newly_exposed] = t
        
        #Careful!!!! Need to execute this before adding newly infected to I to avoid logical errors
        for i in I[:]:
            if recovery_rate>random.random():
                I.remove(i)
                R.append(i)
                len_I -= 1
                len_R += 1
        
        #go through all newly exposed and move them from S to I
        for newly_exposed in dict_newly_exposed: 
            try:
                S.remove(newly_exposed)
                len_S -= 1
            except ValueError: #happens if newly_exposed is not in S but in VS
                VS.remove(newly_exposed)
                len_VS -= 1
            I.append(newly_exposed)
            nextstage[newly_exposed]='R'
            len_I += 1


        if OUTPUT:
            res.append([len_VI,len_VS,len_S,len_I,len_R])

        if (len_I+len_R)/N>0.01:
            OUTBREAK=True
            if initial_exposed not in I:#have to wait for this for accurate R0 calculation
                break  
        elif len_I==0:
            if OUTPUT:
                print('stopped after %i iterations because there were no more infected' % t)
            break
        
    if OUTPUT:
        return res,seed,counter_secondary_infections,time_infected,disease_generation_time,tew,tew_spread
    else:
        return len_VI,len_VS,len_S,len_I,len_R,OUTBREAK,actual_clustering_vacc,actual_clustering_dist,actual_correlation,seed,counter_secondary_infections[initial_exposed],np.nanmean(disease_generation_time),np.nanmean(time_infected),tew,tew_spread




#fixed parameters
#N,k,p_edgechange (and nsim) as command line arguments

network_generating_function = nx.watts_strogatz_graph #can theoretically also use nx.newman_watts_strogatz_graph but then there is not the average same number of private and public interactions
initial_seed = np.random.randint(1,2**32 - 1)

neverinfected_counts = []
outbreaks = []
actual_clustering_vaccs = []
actual_clustering_dists = []
actual_correlations = []
R0s = []
mean_generation_times = []
mean_time_infections = []
tews=[]
tew_spreads = []

args = []
for i in np.arange(nsim):
    #randomly sampled from parameter space
    beta = 0.1
    recovery_rate = 0.1
    
    INITIALIZE_EQUALLY = False#random.choice([True,False])

    private_activity_S = 1/np.sqrt(2)#np.random.uniform(0,1)**(1/2)#change of private behavior for ppl in SEA: 1=no change in behavior, 0=all contacts stopped 
    activity_reduction_I = 0#np.random.uniform(0,1) #change of public behavior for ppl in I: 0=no change in behavior, 1=all contacts stopped 
    
    proportion_vaccinated = np.random.uniform(0.2,0.8)
    vaccine_efficacy	 = np.random.uniform(0,1)
    proportion_distancers = np.random.uniform(0.2,0.8)#np.random.uniform(0.5,0.8)
    activity_reduction_distancers = np.random.uniform(0,1)
    #private_activity_V = private_activity_S#np.random.uniform(private_activity_S,1)#
    #private_activity_V = random.choice(np.linspace(private_activity_S,1,3))#np.random.uniform(private_activity_S,1)#
    #private_activity_V = np.random.uniform(private_activity_S,1)#
    private_activity_V = private_activity_S#random.choice([private_activity_S,private_activity_S*1.1,private_activity_S*1.2,private_activity_S*1.3,private_activity_S*1.4])#np.random.uniform(private_activity_S,1)#
    
    expected_clustering_v = proportion_vaccinated**2+(1-proportion_vaccinated)**2
    #clustering_vaccine = random.choice([expected_clustering_v+el*(1-expected_clusstering_v) for el in [0,0.25,0.5]])
    #clustering_vaccine = np.random.uniform(expected_clustering_v,expected_clustering_v+0.5*(1-expected_clustering_v))
    #clustering_vaccine = random.choice([expected_clustering_v,expected_clustering_v+0.5*(1-expected_clustering_v)])
    
    expected_clustering_d = proportion_distancers**2+(1-proportion_distancers)**2
    #clustering_distancing = random.choice([expected_clustering_d+el*(1-expected_clustering_d) for el in [0,0.25,0.5]])
    #clustering_distancing = np.random.uniform(expected_clustering_d,expected_clustering_d+0.5*(1-expected_clustering_d))
    #clustering_distancing = random.choice([expected_clustering_d,expected_clustering_d+0.5*(1-expected_clustering_d)])

    #allowable_range_corrrelations = opi.get_allowable_range(proportion_vaccinated,proportion_distancers,15)
    #correlation_opinions = np.random.uniform(0.9*allowable_range_corrrelations[0],0.9*allowable_range_corrrelations[1])
    #minmax = min(-allowable_range_corrrelations[0],allowable_range_corrrelations[1])
    #correlation_opinions = np.random.uniform(-0.9*minmax,0.9*minmax)
    #correlation_opinions = random.choice([-0.45,0,0.45])#random.choice([0.9*allowable_range_corrrelations[0],0,0.4*allowable_range_corrrelations[1]])      
    
    if random.random()>0.5:
        clustering_vaccine=expected_clustering_v
        clustering_distancing=expected_clustering_d
    else:
        clustering_vaccine=expected_clustering_v+0.5*(1-expected_clustering_v)
        clustering_distancing=expected_clustering_d+0.5*(1-expected_clustering_d)
    correlation_opinions=0
    
    
    activity_distancers = 1-activity_reduction_distancers
    while True: #for whatever reason  clustering_v2 sometimes (rarely) causes an index error
        try:
            len_VI,len_VS,len_S,len_I,len_R,outbreak,actual_clustering_vacc,actual_clustering_dist,actual_correlation,seed,R0,mean_generation_time,mean_time_infection,tew,tew_spread = model(N,k,p_edgechange,network_generating_function,T_max,beta,recovery_rate,proportion_vaccinated,vaccine_efficacy,private_activity_S,private_activity_V,activity_reduction_I,activity_distancers,proportion_distancers,clustering_vaccine,clustering_distancing,correlation_opinions,INITIALIZE_EQUALLY,seed=None)
            break
        except IndexError:
            pass
    neverinfected_count = len_S+len_VS+len_VI
    neverinfected_counts.append(neverinfected_count)
    outbreaks.append(outbreak)
    actual_clustering_vaccs.append(actual_clustering_vacc)
    actual_clustering_dists.append(actual_clustering_dist)
    actual_correlations.append(actual_correlation)
    R0s.append(R0)
    mean_generation_times.append(mean_generation_time)
    mean_time_infections.append(mean_time_infection)
    tews.append(tew)
    tew_spreads.append(tew_spread)

    args.append([N,k,p_edgechange,network_generating_function,T_max,beta,recovery_rate,proportion_vaccinated,vaccine_efficacy,private_activity_S,private_activity_V,activity_reduction_I,activity_reduction_distancers,proportion_distancers,clustering_vaccine,clustering_distancing,correlation_opinions,INITIALIZE_EQUALLY,seed])
    #print(i,time.time()-now)
args = np.array(args)
#to get args_names, run this ','.join(["'"+el+"'" for el in '''N,k,p_edgechange,network_generating_function,T_max,beta,recovery_rate,proportion_vaccinated,vaccine_efficacy,private_activity_S,private_activity_V,activity_reduction_I,activity_reduction_distancers,proportion_distancers,clustering_vaccine,clustering_distancing,correlation_opinions,INITIALIZE_EQUALLY,initial_seed'''.split(',')])
args_names = ['N','k','p_edgechange','network_generating_function','T_max','beta','recovery_rate','proportion_vaccinated','vaccine_efficacy','private_activity_S','private_activity_V','activity_reduction_I','activity_reduction_distancers','proportion_distancers','clustering_vaccine','clustering_distancing','correlation_opinions','INITIALIZE_EQUALLY','initial_seed']

f = open(output_folder+'output_model%s_nsim%i_N%i_k%i_seed%i_SLURM_ID%i.txt' % (version,nsim,N,k,initial_seed,SLURM_ID) ,'w')
f.write('filename\t'+filename+'\n')
f.write('SLURM_ID\t'+str(SLURM_ID)+'\n')
for ii in range(len(args_names)):#enumerate(zip(args,args_names)):
    if args_names[ii] in ['network_generating_function'] or len(set(args[:,ii])) == 1:
        f.write(args_names[ii]+'\t'+str(args[0,ii])+'\n')    
    else:
        f.write(args_names[ii]+'\t'+'\t'.join(list(map(str,[el if type(el)!=float else round(el,9) for el in args[:,ii]])))+'\n')

f.write('neverinfected_counts\t'+'\t'.join(list(map(str,neverinfected_counts)))+'\n')
f.write('outbreaks\t'+'\t'.join(list(map(str,outbreaks)))+'\n')
f.write('actual_clustering_vaccination\t'+'\t'.join(list(map(str,actual_clustering_vaccs)))+'\n')
f.write('actual_clustering_distancing\t'+'\t'.join(list(map(str,actual_clustering_dists)))+'\n')
f.write('actual_correlation_vacc_dist\t'+'\t'.join(list(map(str,actual_correlations)))+'\n')
f.write('R0s\t'+'\t'.join(list(map(str,[round(el,3) for el in R0s])))+'\n')
f.write('mean_generation_times\t'+'\t'.join(list(map(str,[round(el,3) for el in mean_generation_times])))+'\n')
f.write('mean_time_infections\t'+'\t'.join(list(map(str,[round(el,3) for el in mean_time_infections])))+'\n')
f.write('average_edge_weight\t'+'\t'.join(list(map(str,[round(el,6) for el in tews])))+'\n')
f.write('average_edge_weight_spread\t'+'\t'.join(list(map(str,[round(el,6) for el in tew_spreads])))+'\n')
f.close()

#print(time.time()-now)


#f,ax = plt.subplots()
#xs = np.arange(1,12)
#ys1 = [1-math.exp(-beta*x) for x in xs]
#ys2 = [1-(1-beta)**x for x in xs]
#ax.plot(xs,ys1,'x--',label=r'$1-e^{-\beta*I}$')
#ax.plot(xs,ys2,'o-',label=r'$1-(1-beta)^I$')
#ax.set_xlabel('Number of infected neighbors, I')
#ax.set_ylabel('Infection probability')
#ax.legend(loc='best')
#plt.savefig('comparison_infection_probability_sebastian_ourmodel.pdf')