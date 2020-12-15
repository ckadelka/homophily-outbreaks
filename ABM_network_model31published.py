#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 18:19:45 2020

@author: ckadelka
"""

'''
00: basic SEIHRD model (H=hospitalized infected, D=dead)
01: added limited hospital size  (limited number of beds/respirators/ICUs)
    by adding a group C of ppl who require hospitalization/care but cannot get it
02: added social distancing between non-vulnerable and vulnerable population
03: added daily shopping trips/group gatherings with random people
04: changes: distributions for transition times, rather than Markov chain-induced geometric distribution,
    at each transition, each individual gets assigned a random time until the next transition
05: when ppl become hospitalized, they have a randomly assigned time until recovery (severity of infection/implications)
    any hospitalized has the same per-day death rate
    when hospitalized don't receive perfect care, the recovery takes longer (each day in the hospital they move <1 to recovery)
06: Shortened parameter names, deleted obsolete code, incluced watts_strogatz_graph networkx-independent, added estimation of R0
07: skipped
08: skipped
09: model implemented as a function, basic plotting added
10: stripped version for parallel computing, added SLURM_ID support, runs different social distancing choices (not same R0 across all options)
11: added four different hospital triaging policies, keeping the total care provided constant across the options
    (that is, the total care provided only depends on the level of overcapacity of the hospital)
12: changed model to (by default) always calculate R0, runs for a random choice in the parameter space
13: generalized formulas so that cases where p_IH_lowrisk!=0 and p_HD_lowrisk!=0 can be considered,
    added a preliminary version of testing in the model (introduced tested as another attribute just like ages), 
    anybody tested positive engages in a strong reduction of all activity but no testing policies yet
14: skipped
15: testing_policy_decision_minor works, testing_policy_decision_major (highrisk vs lowrisk ), not yet
16: (still missing: testing_policy_decision_major (highrisk vs lowrisk ) and reduction_highrisk), works
17: runs a huge number of single simulations, completely uniformly sampling from the parameter space
18: corrected a mistake in the calculation of R0, before in some instances R0==0,deaths>1 could be reported
    because the calculation of R0 was stopped too early (i.e. when patient 0 was still infectious),
D    corrected the way contact probabilities are calculated, before the geometric mean was used to combine the contact
    rate between two people, now we use the arithmetric mean
19: corrected another mistake in the calculation of R0, now it is correct: If R0=0, nobody gets infected, otherwise R0>=1,
    changed the way edge weights are calculated, now using the law of mass action and not sqrt(law of mass action)
20: added recorded of peak number of infected, hospitalized and symptomatic witthout testing
21: reduced the baseline contact rate: the private small-world interaction network still has k average connections per node 
    but each connection (edge) is only active on a given day with a certain probability,
    accordingly adjusted the probability of public interactions to keep same average numbers of private and public interactions
22: deleted the explicit modeling of the rate of contact introduced in v21, instead b_I and b_A implicitly correspond to 
    the transmission rate, which is the rate of contact times the probability of transmission (given contact).
    Changed the transition times from Poisson to continuous distributions found in the literature using a int(round(x)) to discretize
23: k varies modeling low contact countries (k=4, Germany), mediocre (k=7), and high (k=10, Italy)
24: skipped
25: skipped
26: final version of model, varies k (k needs to be even for small-world networks!)
27: same as 26, just 10000 as default network size
28: added vaccination, i.e. a parameter that describes what proportion gets vaccinated, and anothe parameter that describes the efficacy of the vaccine
29: added clustering and correlation of vaccine belief and social distancing
30: added third dimension for clustering and correlation: high-risk vs low-risk (i.e. age)
31: published version
'''

version='31'
ONLY_LOOK_FOR_OUTBREAK = False

#built-in modules
import sys
import random
import math

#added modules
import numpy as np
import networkx as nx
#import itertools
from scipy.interpolate import interp1d
import opinion_formation07 as opi

output_folder = 'results/model%s/' % version
output_folder = ''#'results/model%s/' % version


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
    
if len(sys.argv)>6:
    p_highrisk = float(sys.argv[6]) #proportion of highrisk people, p_lowrisk = 1-p_highrisk
else:
    p_highrisk = 1/3
    
#network_generating_function = nx.newman_watts_strogatz_graph

## parameters for overall simulation
OUTPUT = 0#True
T_max = 1000 #in days

def get_optimal_per_day_death_rate(params_HR,p_HD=0.01,GET_INTERPOLATOR=False):
    #X ~ Poisson(t_HR), Y ~ Geom(d), fit the per day death rate d such that P(X>Y) = p_HD
    t_HR = params_HR if type(params_HR) in [float,int] else params_HR[0]
    fak = 1
    prob_X_is_k = []
    upto = 8*t_HR
    for k in range(upto):
        prob_X_is_k.append( t_HR**k*math.exp(-t_HR)/fak)
        fak*=(k+1)
    ps = np.arange(0.0001,0.2,0.001)
    p_Y_less_than_Xs = []
    for per_day_death_rate_hospital in ps:
        prob_Y_less_than_k = [0]
        for k in range(1,upto):
            prob_Y_less_than_k.append(1 - (1-per_day_death_rate_hospital)**k)
        p_Y_less_than_X = np.dot(prob_X_is_k,prob_Y_less_than_k)
        p_Y_less_than_Xs.append(p_Y_less_than_X)
    p_Y_less_than_Xs = np.array(p_Y_less_than_Xs)
    #np.interp(p_Y_less_than_Xs,ps)
    f2 = interp1d(p_Y_less_than_Xs, ps, kind='cubic')
    if GET_INTERPOLATOR:
        return f2
    else:
        return f2([p_HD])[0]

##Functions modeling random transitions
def get_random_course_of_infection_E_to_A_or_I(p_A,params_EA,params_EI):
    if p_A>random.random():
        nextstage = 'A'
        timeto = max(1,int(round(np.random.poisson(*params_EA))))
    else:
        nextstage = 'I'
        timeto = max(1,int(round(np.random.poisson(*params_EI))))
    return (nextstage,timeto)

def get_random_course_of_infection_I_to_H_or_R(p_H,params_IH,params_IR):
    if p_H>random.random():
        nextstage = 'H'
        timeto = max(1,int(round(np.random.poisson(*params_IH))))
    else:
        nextstage = 'R'
        timeto = max(1,np.random.poisson(params_IR))
    return (nextstage,timeto)

def get_random_time_until_recovery_if_asymptomatic(params_AR):
    timeto = np.random.poisson(params_AR)
    nextstage = 'RA' 
    return (nextstage,timeto)

def get_random_time_until_recovery_under_perfect_care(params_HR):
    timeto = np.random.poisson(params_HR)
    nextstage = 'Unknown' #might die each day in hospital due to hospital-associated per-day death rate
    return (nextstage,timeto)

shape=2
scale=2
transmission_rate_over_time = np.array([t**(shape-1)*np.exp(-t/scale) for t in range(1,50+1)])
normalized_transmission_rate_over_time = transmission_rate_over_time/max(transmission_rate_over_time)

def get_transmission_probs(t_EAI,b_AI,activity_reduction_due_to_age,activity_reduction_due_to_distancing):#could make this expoentially increasing but that requires knowledge of how the viral load increases
    return [0 for t in range(1,int(t_EAI)-1)] + list(activity_reduction_due_to_age*activity_reduction_due_to_distancing*b_AI*normalized_transmission_rate_over_time)

def total_edge_weight_in_susceptible_network(network,private_activity_SEA,private_activity_V,currentstage,activities_distancers,activities_highrisk):
    #c = probability_of_close_contact_with_random_person_in_public    
    total_edge_weight_private = 0
    n_edges = 0
    for nodeA,nodeB in network.edges():
        activityA = (private_activity_SEA if currentstage[nodeA]=='S' else private_activity_V)*activities_distancers[nodeA]*activities_highrisk[nodeA]
        activityB = (private_activity_SEA if currentstage[nodeB]=='S' else private_activity_V)*activities_distancers[nodeB]*activities_highrisk[nodeB]
        total_edge_weight_private +=activityA*activityB
        n_edges+=1

    return total_edge_weight_private/n_edges

def model(N,k,p_edgechange,network_generating_function,p_highrisk,T_max,b_A,b_I,b_H,params_EA,params_EI,params_IH,params_AR,params_IR,params_HR,p_A,p_A_lowrisk_over_p_A_highrisk,p_IH,p_IH_lowrisk_over_p_IH_highrisk,overall_death_rate_covid19,p_HD_lowrisk_over_p_HD_highrisk,probability_of_close_contact_with_random_person_in_public,private_activity_SEA,private_activity_I,private_activity_H,private_highrisk_lowrisk_activity,public_activity_SEA,activity_highrisk,activity_distancers,public_activity_I,public_activity_H,hospital_beds_per_person,care_decline_exponent,triaging_policy,max_number_of_tests_available_per_day,activity_P,testing_policy_decision_major,testing_policy_decision_minor,testing_delay,proportion_vaccinated,vaccine_efficacy,private_activity_V,public_activity_V,proportion_distancers,clustering_vaccine,clustering_distancing,clustering_highrisk,corr_vd,corr_vr,corr_dr,corr_vdr,seed=None,OUTPUT=False,ESTIMATE_R0=True,interpolator_per_day_death_rate=None):
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

    ps = [proportion_vaccinated,proportion_distancers,p_highrisk]
    thresholds = [clustering_vaccine,clustering_distancing,clustering_highrisk]
        
    opinions,nr_swaps = opi.opinion_clustering_3d_v2b(network,ps=ps,c12=corr_vd,c13=corr_vr,c23=corr_dr,c123=corr_vdr,nsim=10*N,thresholds=thresholds,exponent=16)
    actual_clustering_vacc,actual_clustering_dist,actual_clustering_risk = opi.get_proportion_of_edges_with_same_opinion_2d(network,opinions)
    vaccinated = list(opinions[0])
    distancing = list(opinions[1])
    ages = list(opinions[2])
    activities_highrisk = [activity_highrisk if ages[i]==1 else 1 for i in range(N)]
    
    
    actual_corr_vd = np.corrcoef(vaccinated,distancing)[0,1]
    actual_corr_vr = np.corrcoef(vaccinated,ages)[0,1]
    actual_corr_dr = np.corrcoef(distancing,ages)[0,1]

    #standardize storage of parameters for transition time distributions to work for single- and multi-parameter distributions
    params_EA = [params_EA] if type(params_EA) in [float,int] else params_EA #mean of Poisson RV
    params_EI = [params_EI] if type(params_EI) in [float,int] else params_EI #mean of Poisson RV
    params_IH = [params_IH] if type(params_IH) in [float,int] else params_IH #mean of Poisson RV
    params_AR = [params_AR] if type(params_AR) in [float,int] else params_AR #mean of Poisson RV
    params_IR = [params_IR] if type(params_IR) in [float,int] else params_IR #mean of Poisson RV
    params_HR = [params_HR] if type(params_HR) in [float,int] else params_HR #mean of Poisson RV

    nextstage = ['E' for _ in range(N)]
    currentstage = ['S' for _ in range(N)]
    time_to_nextstage = np.array([-1 for _ in range(N)],dtype=float)
    OUTBREAK = False

    #derive transition probabilities based on differential risk ratios
    p_A_highrisk = p_A/(p_A_lowrisk_over_p_A_highrisk*(1-p_highrisk)+p_highrisk) #calculate the probability that a highrisk exposed person will have an asymptomatic infection
    p_A_lowrisk = p_A_lowrisk_over_p_A_highrisk*p_A_highrisk #calculate the probability that a lowrisk  exposed person will have an asymptomatic infection
    p_IH_highrisk = (1-p_A)*p_IH/(  p_highrisk*(1-p_A_highrisk) + p_IH_lowrisk_over_p_IH_highrisk*(1-p_highrisk)*(1-p_A_lowrisk)  )
    p_IH_lowrisk = p_IH_lowrisk_over_p_IH_highrisk*p_IH_highrisk 
    p_HD = overall_death_rate_covid19/(p_highrisk*(1-p_A_highrisk)*p_IH_highrisk + (1-p_highrisk)*(1-p_A_lowrisk)*p_IH_lowrisk)
    p_HD_highrisk = (1-p_A)*p_IH*p_HD/(  p_highrisk*(1-p_A_highrisk)*p_IH_highrisk + p_HD_lowrisk_over_p_HD_highrisk*(1-p_highrisk)*(1-p_A_lowrisk)*p_IH_lowrisk  )
    p_HD_lowrisk = p_HD_lowrisk_over_p_HD_highrisk*p_HD_highrisk 

    if interpolator_per_day_death_rate!=None and interpolator_per_day_death_rate[0]==params_HR:
        pass
    else:
        interpolator_per_day_death_rate = [params_HR,get_optimal_per_day_death_rate(params_HR,0.01,True)]
    per_day_death_rate_hospital_highrisk = interpolator_per_day_death_rate[1]([p_HD_highrisk])[0] #Bernoulli (death in hospital is geometrically distributed, the longer one stays in the hospital the more likely to die)
    per_day_death_rate_hospital_lowrisk = interpolator_per_day_death_rate[1]([p_HD_lowrisk])[0] #Bernoulli (death in hospital is geometrically distributed, the longer one stays in the hospital the more likely to die)

    #S-E-I-R or S-E-I-H-R or S-E-I-H-D
    #S=susceptible, E=exposed, A=asymptomatic, I=symptomatic, H=severely infected, requiring hospitalization, R=recovered, D=dead    
    
    S = []
    VS = []#vaccinated but susceptible
    VI = []#vaccinated and immune   
    for el in range(N):
        if vaccinated[el]:
            if random.random()<vaccine_efficacy:
                VI.append(el)
                nextstage[el]='VS' #eventually (in future version of the model, the vaccine response may wane over time, at which point the next stage is VS)
            else:
                VS.append(el)
        else:
            S.append(el)
    
    activities_distancers = [activity_distancers if distancing[i] else 1 for i in range(N)]

    #calculate initial total edge weight = 1 - overall contact reduction
    total_edge_weight = total_edge_weight_in_susceptible_network(network,private_activity_SEA,private_activity_V,currentstage,activities_distancers,activities_highrisk)

    initial_exposed = random.choices(S+VS,weights=[activities_distancers[el] for el in S+VS])[0]
    
    p_A = p_A_highrisk if ages[initial_exposed]==1 else p_A_lowrisk
    nextstage[initial_exposed],time_to_nextstage[initial_exposed] = get_random_course_of_infection_E_to_A_or_I(p_A,params_EA,params_EI)
    currentstage[initial_exposed] = 'E'
    dict_transmission_probs = dict({initial_exposed:get_transmission_probs(time_to_nextstage[initial_exposed],b_I if nextstage[initial_exposed]=='I' else b_A,activity_highrisk if ages[initial_exposed]==1 else 1,activity_distancers if distancing[initial_exposed] else 1)})
    E = [initial_exposed]
    A = []
    I = []
    R = [] #R=RP, recovered not in RA were symptomatic and assume that they had COVID19
    RA = []
    RP = []
    H = [] 
    D = []        

    len_E_highrisk = 1 if ages[initial_exposed]==1 else 0
    len_A_highrisk = 0
    len_I_highrisk = 0
    len_H_highrisk = 0
    len_E_lowrisk = 1 if ages[initial_exposed]==0 else 0
    len_A_lowrisk = 0
    len_I_lowrisk = 0
    len_H_lowrisk = 0
    
    time_infected = [np.nan for i in range(N)]
    time_infected[initial_exposed] = 0
    disease_generation_time = [np.nan for i in range(N)]
    infections_caused_byE,infections_caused_byA,infections_caused_byI = 0,0,0


    #initially nobody is tested
    tested_positive = np.zeros(N,dtype=bool)
    time_to_test_result = [-1 for i in range(N)]
    
    #initially nobody will be tested
    TEST=False

    I_not_tested_highrisk = []
    I_not_tested_lowrisk = []
    H_not_tested_highrisk = []
    H_not_tested_lowrisk = []
    len_I_not_tested_highrisk = len(I_not_tested_highrisk)
    len_I_not_tested_lowrisk = len(I_not_tested_lowrisk)
    len_H_not_tested_highrisk = len(H_not_tested_highrisk)
    len_H_not_tested_lowrisk = len(H_not_tested_lowrisk)
    I_waiting = []
    H_waiting = []
    len_IP_highrisk=0
    len_HP_highrisk=0    
    len_AP_highrisk=0
    len_EP_highrisk=0
    len_IP_lowrisk=0
    len_HP_lowrisk=0    
    len_AP_lowrisk=0
    len_EP_lowrisk=0

    max_len_H,max_len_I,max_len_not_tested = 0,0,0
    
    if triaging_policy in ['FBLS','FBrnd']:
        currently_in_care = []

    if OUTPUT:
        res = [[len(S),len(E),len(A),len(I),len(H),len(R),len(RA),len(RP),len(D)]]
        
    if ESTIMATE_R0:
        counter_secondary_infections = [0 for i in range(N)]
    else:
        counter_secondary_infections = [np.nan for i in range(N)]
    
    for t in range(1,T_max+1):
        #things that happen on any day (synchronously):
        #1. All S can get infected via private or public interactions,
        #   this happens with a certain probability based on activity levels (social distancing policies)
        #2. All newly infected S move to E,
        #   and it is determined (Bernoulli random variable) if they continue to move to A or I as well as the next transition time (Poisson)
        #3. All E, I, A move one day "closer" to the next compartment (E->I/A, I->H/R, A->RA), 
        #   if they "reach" the next compartment, a Poisson-distributed transition time to the next compartment and,
        #   possibly, a random Bernoulli variable deciding which compartment is drawn
        #4. All H have a risk of dying (Bernoulli random variable), 
        #5. All H move closer to R (the "distance" they move closer depends on the triaging_policy and hospital overcapacity),
        #   If they "reach" R, they move to R and are recovered.
        #   Note: The risk of dying is proportionally reduced for individuals that don't require a full day for recovery.
        #6. Any person may get tested (testing policies decide who gets limited tests).
        #7. Any person may receive a test result (possibly delayed),
        #   and if positive, move to a special category (E->EP, A->AP, I->IP, H->HP, R->RP). 
        #   Positive tests significantly reduce activity levels, both public and private, due to quarantine, however not to 0 due to imperfect quarantine measures taken by the average person
        #   Note: There is no category SP because FPR is assumed to be 0. 
        #   Further, there are no negative test categories because a negative test "today" does not exclude a positive test "tomorrow".

        
        #start testing once the first sympytomatic person presents at the hospital
        if len_I_highrisk+len_I_lowrisk>0 and TEST==False:
            TEST=True
        
        if TEST:
            assert testing_policy_decision_minor in ['random','FIFT','LIFT']
            assert testing_policy_decision_major in ['O>Y','Y>O']
            #pick who to test based on policies, need to know for which category who has not been tested,
            #test results get back at the earliest at the end of day (if delay = 0) so on the day of testing
            #tested individuals can still infect others
            tests_left = max_number_of_tests_available_per_day
            order_H,order_I = 0,1
            order_highrisk,order_lowrisk = (0,1) if testing_policy_decision_major=='O>Y' else (1,0)
            counter_major = 0
            counter_minor = 0
            while tests_left>0 and (len_H_not_tested_highrisk>0 or len_I_not_tested_highrisk>0 or len_H_not_tested_lowrisk>0 or len_I_not_tested_lowrisk>0):
                if len_H_not_tested_highrisk>0 and counter_minor==order_H and counter_major==order_highrisk:
                    if testing_policy_decision_minor =='random':
                        h = H_not_tested_highrisk.pop(int(random.random()*len_H_not_tested_highrisk))
                    elif testing_policy_decision_minor =='FIFT':
                        h = H_not_tested_highrisk.pop(0)
                    elif testing_policy_decision_minor =='LIFT':
                        h = H_not_tested_highrisk.pop()
                    time_to_test_result[h] = testing_delay+1
                    len_H_not_tested_highrisk -= 1
                    H_waiting.append(h)
                    tests_left -= 1
                elif len_H_not_tested_highrisk==0 and counter_minor==order_H and counter_major==order_highrisk: #ran out of H to be tested
                    counter_minor+=1
                elif len_H_not_tested_lowrisk>0 and counter_minor==order_H and counter_major==order_lowrisk:
                    if testing_policy_decision_minor =='random':
                        h = H_not_tested_lowrisk.pop(int(random.random()*len_H_not_tested_lowrisk))
                    elif testing_policy_decision_minor =='FIFT':
                        h = H_not_tested_lowrisk.pop(0)
                    elif testing_policy_decision_minor =='LIFT':
                        h = H_not_tested_lowrisk.pop()
                    time_to_test_result[h] = testing_delay+1
                    len_H_not_tested_lowrisk -= 1
                    H_waiting.append(h)
                    tests_left -= 1
                elif len_H_not_tested_lowrisk==0 and counter_minor==order_H and counter_major==order_lowrisk: #ran out of H to be tested
                    counter_minor+=1             
                elif len_I_not_tested_highrisk>0 and counter_minor==order_I and counter_major==order_highrisk:
                    if testing_policy_decision_minor =='random':
                        i = I_not_tested_highrisk.pop(int(random.random()*len_I_not_tested_highrisk))
                    elif testing_policy_decision_minor =='FIFT':
                        i = I_not_tested_highrisk.pop(0)
                    elif testing_policy_decision_minor =='LIFT':
                        i = I_not_tested_highrisk.pop()
                    time_to_test_result[i] = testing_delay+1
                    len_I_not_tested_highrisk -= 1
                    I_waiting.append(i)
                    tests_left -= 1
                elif len_I_not_tested_highrisk==0 and counter_minor==order_I and counter_major==order_highrisk: #ran out of I to be tested
                    counter_minor+=1
                elif len_I_not_tested_lowrisk>0 and counter_minor==order_I and counter_major==order_lowrisk:
                    if testing_policy_decision_minor =='random':
                        i = I_not_tested_lowrisk.pop(int(random.random()*len_I_not_tested_lowrisk))
                    elif testing_policy_decision_minor =='FIFT':
                        i = I_not_tested_lowrisk.pop(0)
                    elif testing_policy_decision_minor =='LIFT':
                        i = I_not_tested_lowrisk.pop()
                    time_to_test_result[i] = testing_delay+1
                    len_I_not_tested_lowrisk -= 1
                    I_waiting.append(i)
                    tests_left -= 1
                elif len_I_not_tested_lowrisk==0 and counter_minor==order_I and counter_major==order_lowrisk: #ran out of I to be tested
                    counter_minor+=1  
                if counter_minor==2:
                    counter_minor = 0
                    counter_major += 1
                    

        if TEST:
            #see who gets test results back
            for i in I_waiting[:]:
                time_to_test_result[i] -= 1
                if time_to_test_result[i]==0: #test results come back
                    dict_transmission_probs[i] = [activity_P*el for el in dict_transmission_probs[i]]
                    I_waiting.remove(i)
                    tested_positive[i]=True
                    if ages[i]==1:
                        len_IP_highrisk+=1
                    else:
                        len_IP_lowrisk+=1                        
            for h in H_waiting[:]:
                time_to_test_result[h] -= 1
                if time_to_test_result[h]==0: #test results come back
                    dict_transmission_probs[h] = [activity_P*el for el in dict_transmission_probs[h]]
                    H_waiting.remove(h)
                    tested_positive[h]=True
                    if ages[h]==1:
                        len_HP_highrisk+=1
                    else:
                        len_HP_lowrisk+=1                     
        
        
        #go through all exposed, asymptomatic, symptomatic and hospitalized and check whether they cause new infections among their private or random public contacts
        dict_newly_exposed = {}

        prob_public_infection_highrisk_S = []
        prob_public_infection_lowrisk_S = []
        prob_public_infection_highrisk_VS = []
        prob_public_infection_lowrisk_VS = []
        prob_public_infection_highrisk_distancing_S = []
        prob_public_infection_lowrisk_distancing_S = []
        prob_public_infection_highrisk_distancing_VS = []
        prob_public_infection_lowrisk_distancing_VS = []
        for ii,(contagious_compartment,private_activity_level_contagious,public_activity_level_contagious) in enumerate(zip([E,A,I,H],[private_activity_SEA,private_activity_SEA,private_activity_I,private_activity_H],[public_activity_SEA,public_activity_SEA,public_activity_I,public_activity_H])):
            for contagious in contagious_compartment:
                try:
                    current_transmission_prob = dict_transmission_probs[contagious].pop(0)
                except IndexError: #same patients can stay really long in a potentially contagious category, especially H, set infectivity to 0 then.
                    current_transmission_prob = 0
                private_activity_times_infectiousness = current_transmission_prob * (private_activity_level_contagious if (ii>1 or vaccinated[contagious]==False) else private_activity_V)
                if private_activity_level_contagious>0:
                    for neighbor in neighbors[contagious]:
                        #private/local contacts:
                        if nextstage[neighbor]=='E':
                            probability_of_infection = private_activity_times_infectiousness * (private_activity_SEA if currentstage[neighbor]=='S' else private_activity_V) * activities_highrisk[neighbor] * activities_distancers[neighbor] * (private_highrisk_lowrisk_activity if ages[neighbor]!=ages[contagious] else 1)
                            if probability_of_infection>random.random():                           
                                try:
                                    dict_newly_exposed[neighbor].append(contagious)
                                except KeyError:
                                    dict_newly_exposed.update({neighbor:[contagious]})
                #public/random contacts:
                if public_activity_SEA>0:
                    public_activity_times_infectiousness = current_transmission_prob * (public_activity_level_contagious if (ii>1 or vaccinated[contagious]==False) else public_activity_V)
                    prob_public_infection_lowrisk_S.append(probability_of_close_contact_with_random_person_in_public*public_activity_times_infectiousness*(public_activity_SEA))
                    prob_public_infection_highrisk_S.append(probability_of_close_contact_with_random_person_in_public*public_activity_times_infectiousness*(public_activity_SEA*activity_highrisk))
                    prob_public_infection_lowrisk_VS.append(probability_of_close_contact_with_random_person_in_public*public_activity_times_infectiousness*(public_activity_V))
                    prob_public_infection_highrisk_VS.append(probability_of_close_contact_with_random_person_in_public*public_activity_times_infectiousness*(public_activity_V*activity_highrisk))
                    prob_public_infection_lowrisk_distancing_S.append(probability_of_close_contact_with_random_person_in_public*public_activity_times_infectiousness*(public_activity_SEA*activity_distancers))
                    prob_public_infection_highrisk_distancing_S.append(probability_of_close_contact_with_random_person_in_public*public_activity_times_infectiousness*(public_activity_SEA*activity_highrisk*activity_distancers))
                    prob_public_infection_lowrisk_distancing_VS.append(probability_of_close_contact_with_random_person_in_public*public_activity_times_infectiousness*(public_activity_V*activity_distancers))
                    prob_public_infection_highrisk_distancing_VS.append(probability_of_close_contact_with_random_person_in_public*public_activity_times_infectiousness*(public_activity_V*activity_highrisk*activity_distancers))
        
        if public_activity_SEA>0:
            probability_that_lowrisk_S_gets_infected_publicy = 1-np.prod([1-el for el in prob_public_infection_lowrisk_S])
            probability_that_highrisk_S_gets_infected_publicy = 1-np.prod([1-el for el in prob_public_infection_highrisk_S])
            probability_that_lowrisk_VS_gets_infected_publicy = 1-np.prod([1-el for el in prob_public_infection_lowrisk_VS])
            probability_that_highrisk_VS_gets_infected_publicy = 1-np.prod([1-el for el in prob_public_infection_highrisk_VS])
            probability_that_lowrisk_distancing_S_gets_infected_publicy = 1-np.prod([1-el for el in prob_public_infection_lowrisk_distancing_S])
            probability_that_highrisk_distancing_S_gets_infected_publicy = 1-np.prod([1-el for el in prob_public_infection_highrisk_distancing_S])
            probability_that_lowrisk_distancing_VS_gets_infected_publicy = 1-np.prod([1-el for el in prob_public_infection_lowrisk_distancing_VS])
            probability_that_highrisk_distancing_VS_gets_infected_publicy = 1-np.prod([1-el for el in prob_public_infection_highrisk_distancing_VS])
    
    
            list_all_contagious = E+A+I+H
                    
            for s in S:
                if ages[s]==1 and distancing[s] and probability_that_highrisk_distancing_S_gets_infected_publicy>random.random():
                    who_did_it = random.choices(list_all_contagious,weights=prob_public_infection_highrisk_distancing_S)[0]
                    try:
                        dict_newly_exposed[s].append(who_did_it)
                    except KeyError:
                        dict_newly_exposed.update({s:[who_did_it]})
                elif ages[s]==1 and distancing[s]==False and probability_that_highrisk_S_gets_infected_publicy>random.random():
                    who_did_it = random.choices(list_all_contagious,weights=prob_public_infection_highrisk_S)[0]
                    try:
                        dict_newly_exposed[s].append(who_did_it)
                    except KeyError:
                        dict_newly_exposed.update({s:[who_did_it]})
                elif ages[s]==0 and distancing[s] and probability_that_lowrisk_distancing_S_gets_infected_publicy>random.random():
                    who_did_it = random.choices(list_all_contagious,weights=prob_public_infection_lowrisk_distancing_S)[0]
                    try:
                        dict_newly_exposed[s].append(who_did_it)
                    except KeyError:
                        dict_newly_exposed.update({s:[who_did_it]})                                                 
                elif ages[s]==0 and distancing[s]==False and probability_that_lowrisk_S_gets_infected_publicy>random.random():
                    who_did_it = random.choices(list_all_contagious,weights=prob_public_infection_lowrisk_S)[0]
                    try:
                        dict_newly_exposed[s].append(who_did_it)
                    except KeyError:
                        dict_newly_exposed.update({s:[who_did_it]}) 
                        
            for s in VS:
                if ages[s]==1 and distancing[s] and probability_that_highrisk_distancing_VS_gets_infected_publicy>random.random():
                    who_did_it = random.choices(list_all_contagious,weights=prob_public_infection_highrisk_distancing_VS)[0]
                    try:
                        dict_newly_exposed[s].append(who_did_it)
                    except KeyError:
                        dict_newly_exposed.update({s:[who_did_it]})
                elif ages[s]==1 and distancing[s]==False and probability_that_highrisk_VS_gets_infected_publicy>random.random():
                    who_did_it = random.choices(list_all_contagious,weights=prob_public_infection_highrisk_VS)[0]
                    try:
                        dict_newly_exposed[s].append(who_did_it)
                    except KeyError:
                        dict_newly_exposed.update({s:[who_did_it]})
                elif ages[s]==0 and distancing[s] and probability_that_lowrisk_distancing_VS_gets_infected_publicy>random.random():
                    who_did_it = random.choices(list_all_contagious,weights=prob_public_infection_lowrisk_distancing_VS)[0]
                    try:
                        dict_newly_exposed[s].append(who_did_it)
                    except KeyError:
                        dict_newly_exposed.update({s:[who_did_it]})                                                 
                elif ages[s]==0 and distancing[s]==False and probability_that_lowrisk_VS_gets_infected_publicy>random.random():
                    who_did_it = random.choices(list_all_contagious,weights=prob_public_infection_lowrisk_VS)[0]
                    try:
                        dict_newly_exposed[s].append(who_did_it)
                    except KeyError:
                        dict_newly_exposed.update({s:[who_did_it]}) 
                        
        #hospitalized get perfect care (progress 1 day closer to recovery) if hospitals aren't overrun, if overrun the care received decreases based on triage policies
        len_H = len_H_highrisk+len_H_lowrisk
        capacity = len_H/N/hospital_beds_per_person# 1= at capacity, 0=compeltely empty, 2=200%, 3=300%
        avg_care_available_per_person = min(1,capacity**(-care_decline_exponent)) if capacity>0 else 1
        total_care_provided = avg_care_available_per_person*len_H
        
        assert triaging_policy in ['FCFS','same','LSF','FBLS','FBrnd']
        if triaging_policy=='same':
            total_care_left = total_care_provided
            nr_without_care_thus_far = len_H
            without_care_thus_far = [True for _ in range(len_H)]
            care_provided_for_each_person = [0 for _ in range(len_H)]
            while total_care_left>0 and nr_without_care_thus_far>0:
                avg_additional_care_available_per_person = total_care_left/nr_without_care_thus_far
                for ii,h in enumerate(H):
                    if without_care_thus_far[ii]:
                        if care_provided_for_each_person[ii]+avg_additional_care_available_per_person>=time_to_nextstage[h]:
                            care_provided_for_each_person[ii] = time_to_nextstage[h]
                            nr_without_care_thus_far-=1
                            without_care_thus_far[ii]=False
                        else:
                            care_provided_for_each_person[ii] += avg_additional_care_available_per_person
                total_care_distributed = sum(care_provided_for_each_person)
                total_care_left = total_care_provided-total_care_distributed
        else: #i.e., if 'FCFS','LSF','FBLS'
            if triaging_policy=='LSF':
                time_to_recovery_under_optimal_care = [time_to_nextstage[h] for h in H]
                #resort H
                H = [h for _,h in sorted(zip(time_to_recovery_under_optimal_care,H))]
            elif triaging_policy=='FBLS': #currently_in_care is a stack
                time_to_recovery_under_optimal_care = [time_to_nextstage[h] for h in H]
                nr_currently_in_care = len(currently_in_care)
                total_care_required_by_those_in_care = sum([min(1,time_to_nextstage[h]) for h in currently_in_care])
                if total_care_required_by_those_in_care>total_care_provided:
                    #resort those currently in care based on severity of infection
                    currently_in_care = [h for _,h in sorted(zip(time_to_recovery_under_optimal_care[:nr_currently_in_care],H[:nr_currently_in_care]))]
                    H = currently_in_care + H[nr_currently_in_care:]            
                else:
                    #resort H so that those who are currently in care are at the front and the remaining ones are sorted based on severity of infection
                    remainder_of_H_sorted = [h for _,h in sorted(zip(time_to_recovery_under_optimal_care,H[nr_currently_in_care:]))]
                    H = currently_in_care + remainder_of_H_sorted
            elif triaging_policy=='FBrnd': #currently_in_care is a stack
                nr_currently_in_care = len(currently_in_care)
                not_in_care = H[nr_currently_in_care:] 
                random.shuffle(not_in_care) #shuffles in place
                H = currently_in_care + not_in_care
            care_provided_for_each_person = []
            total_care_left = total_care_provided
            for h in H:
                if total_care_left>=1:
                    care_for_this_person = min(1,time_to_nextstage[h])
                elif total_care_left==0:
                    care_for_this_person = 0
                else:
                    care_for_this_person = min(total_care_left,time_to_nextstage[h])
                total_care_left -= care_for_this_person
                care_provided_for_each_person.append(care_for_this_person)                
        

        #disease generation time and who caused infections
        for newly_exposed in dict_newly_exposed: 
            len_dict_newly_exposed = len(dict_newly_exposed[newly_exposed])
            for contagious in dict_newly_exposed[newly_exposed]:
                counter_secondary_infections[contagious] += 1/len_dict_newly_exposed
                if currentstage[contagious]=='E':
                    infections_caused_byE += 1/len_dict_newly_exposed
                elif currentstage[contagious]=='A':
                    infections_caused_byA += 1/len_dict_newly_exposed
                elif currentstage[contagious]=='I':
                    infections_caused_byI += 1/len_dict_newly_exposed
                
            if len_dict_newly_exposed>1:
                dummy = []
                for el in dict_newly_exposed[newly_exposed]:
                    dummy.append(t-time_infected[el])
                disease_generation_time[newly_exposed] = np.mean(dummy)
            else:
                disease_generation_time[newly_exposed] = t-time_infected[dict_newly_exposed[newly_exposed][0]]
                time_infected[newly_exposed] = t
                       
        
        #see if hospitalized die (do this simultaneously while transitioning closer to recovery based on care received
        #and checking for recovery), 
        #also: reduce death rate for those that recover that day
        for ii,h in enumerate(H[:]):
            time_left_until_recovery = time_to_nextstage[h]-care_provided_for_each_person[ii]
            part_of_day_that_care_is_required_until_recovery = 1 if time_left_until_recovery>0 else care_provided_for_each_person[ii]
            per_day_death_rate_hospital = per_day_death_rate_hospital_highrisk if ages[h]==1 else per_day_death_rate_hospital_lowrisk
            if per_day_death_rate_hospital*part_of_day_that_care_is_required_until_recovery>random.random():#hospitalized die at certain rate
                nextstage[h] = 'D'
                currentstage[h] = 'D'
                H.remove(h)
                D.append(h)
                time_to_nextstage[h]=t #keeps track of when final stage was reached
                if tested_positive[h]:
                    if ages[h]==1:
                        len_HP_highrisk-=1
                    else:
                        len_HP_lowrisk-=1
                else:
                    try: #if dying person was waiting for test result
                        H_waiting.remove(h)
                        tested_positive[h]=True
                    except ValueError:
                        if ages[h]==1:#highrisk
                            len_H_not_tested_highrisk -= 1
                            H_not_tested_highrisk.remove(h)
                        else:
                            len_H_not_tested_lowrisk -= 1
                            H_not_tested_lowrisk.remove(h)
                if ages[h]==1:
                    len_H_highrisk-=1
                else:
                    len_H_lowrisk-=1 
                if triaging_policy in ['FBLS','FBrnd']:
                    try:
                        currently_in_care.remove(h)
                    except ValueError:#patient that died never received hospital care due to overcapacity
                        pass            
              #hospitalized get better at certain rate, dependent of care level  
            elif time_left_until_recovery<=0:
                H.remove(h)
                currentstage[h]='R'
                nextstage[h] = 'R'
                if ages[h]==1:
                    len_H_highrisk-=1
                else:
                    len_H_lowrisk-=1                    
                time_to_nextstage[h] = t #set this to t to have a counter that keeps track of when recovery happened
                if tested_positive[h]:
                    if ages[h]==1:
                        len_HP_highrisk-=1
                    else:
                        len_HP_lowrisk-=1                        
                    RP.append(h)
                    
                else:
                    try: #if recovered person was waiting for test result
                        H_waiting.remove(h)
                        RP.append(h)
                        tested_positive[h]=True
                    except ValueError:
                        R.append(h)
                        if ages[h]==1:#highrisk
                            len_H_not_tested_highrisk -= 1
                            H_not_tested_highrisk.remove(h)
                        else:
                            len_H_not_tested_lowrisk -= 1
                            H_not_tested_lowrisk.remove(h)
            else:
                time_to_nextstage[h] = max(0,time_left_until_recovery)
                    
                    
        #transition one day closer towards next stage
        for index in E+A+I:
            time_to_nextstage[index] = max(0,time_to_nextstage[index]-1)
        
        #Careful!!!! Need to execute this going from last stage to first stage to avoid logical errors
        for i in I[:]:
            if time_to_nextstage[i]==0:
                if ages[i]==1:
                    len_I_highrisk-=1
                else:
                    len_I_lowrisk-=1                    
                if nextstage[i]=='H':
                    currentstage[i]='H'
                    I.remove(i)
                    H.append(i)
                    nextstage[i],time_to_nextstage[i] = get_random_time_until_recovery_under_perfect_care(params_HR)
                    if tested_positive[i]:
                        if ages[i]==1:
                            len_IP_highrisk-=1
                            len_HP_highrisk+=1
                        else:
                            len_IP_lowrisk-=1
                            len_HP_lowrisk+=1                            
                    else:
                        try: #if hospitalized person was already waiting for test result
                            I_waiting.remove(i)
                            H_waiting.append(i)
                        except ValueError:
                            if ages[i]==1:#hightisk
                                len_I_not_tested_highrisk -= 1
                                len_H_not_tested_highrisk += 1
                                H_not_tested_highrisk.append(i)
                                I_not_tested_highrisk.remove(i) 
                            else:
                                len_I_not_tested_lowrisk -= 1
                                len_H_not_tested_lowrisk += 1
                                H_not_tested_lowrisk.append(i)
                                I_not_tested_lowrisk.remove(i)                                 
                    if ages[i]==1:
                        len_H_highrisk+=1
                    else:
                        len_H_lowrisk+=1                        
                elif nextstage[i]=='R':  
                    currentstage[i]='R'
                    I.remove(i)
                    if tested_positive[i]:
                        if ages[i]==1:
                            len_IP_highrisk-=1
                        else:
                            len_IP_lowrisk-=1                            
                        RP.append(i)
                    else:
                        try: #if hospitalized person was already waiting for test result
                            I_waiting.remove(i)
                            RP.append(i)
                            tested_positive[i]=True
                        except ValueError:
                            R.append(i)
                            if ages[i]==1:#highrisk
                                len_I_not_tested_highrisk -= 1
                                I_not_tested_highrisk.remove(i)
                            else:
                                len_I_not_tested_lowrisk -= 1
                                I_not_tested_lowrisk.remove(i)                                
                    time_to_nextstage[i] = t #set this to t to have a counter that keeps track of when recovery happened
                  
        for a in A[:]:
            if time_to_nextstage[a]==0:  
                A.remove(a)
                if tested_positive[a]:#if tested positive, this person knew they had COVID19 so they go into R, otherwise RA (might get tested randomly if/when considering testing of general population)
                    if ages[a]==1:
                        len_AP_highrisk-=1
                    else:
                        len_AP_lowrisk-=1
                    RP.append(a)
                    currentstage[a]='RP'
                else:
                    RA.append(a)
                    currentstage[a]='RA'
                if ages[a]==1:
                    len_A_highrisk-=1
                else:
                    len_A_lowrisk-=1
                time_to_nextstage[a] = t #set this to t to have a counter that keeps track of when recovery happened

        for e in E[:]: #check for changes, i.e. for time_to_nextstage[e]==0  
            if time_to_nextstage[e]==0:
                if nextstage[e]=='I': #current stage must be E
                    currentstage[e]='I'
                    E.remove(e)
                    I.append(e) 
                    if tested_positive[e]:
                        if ages[e]==1:
                            len_EP_highrisk-=1
                            len_IP_highrisk+=1
                        else:
                            len_EP_lowrisk-=1
                            len_IP_lowrisk+=1                            
                    else:
                        if ages[e]==1:#highrisk
                            I_not_tested_highrisk.append(e)
                            len_I_not_tested_highrisk+=1
                        else:
                            I_not_tested_lowrisk.append(e)
                            len_I_not_tested_lowrisk+=1 
                    if ages[e]==1:
                        len_E_highrisk-=1
                        len_I_highrisk+=1
                    else:
                        len_E_lowrisk-=1
                        len_I_lowrisk+=1                        
                    p_IH = p_IH_highrisk if ages[e]==1 else p_IH_lowrisk
                    nextstage[e],time_to_nextstage[e] = get_random_course_of_infection_I_to_H_or_R(p_IH,params_IH,params_IR)
                elif nextstage[e]=='A': #current stage must be E
                    currentstage[e]='A'
                    E.remove(e)
                    A.append(e)
                    if tested_positive[e]:
                        if ages[e]==1:
                            len_EP_highrisk-=1
                            len_AP_highrisk+=1
                        else:
                            len_EP_lowrisk-=1
                            len_AP_lowrisk+=1 
                    if ages[e]==1:
                        len_E_highrisk-=1
                        len_A_highrisk+=1
                    else:
                        len_E_lowrisk-=1
                        len_A_lowrisk+=1                        
                    nextstage[e],time_to_nextstage[e] = get_random_time_until_recovery_if_asymptomatic(params_AR)
                    
        #go through all newly exposed and move them from S to E, and randomly pick their next stage
        for newly_exposed in dict_newly_exposed: 
            try:
                S.remove(newly_exposed)
            except ValueError: #happens if newly_exposed is not in S but in VS
                VS.remove(newly_exposed)
            E.append(newly_exposed)
            currentstage[newly_exposed]='E'
            if ages[newly_exposed]==1:
                len_E_highrisk+=1
            else:
                len_E_lowrisk+=1                
            p_A = p_A_highrisk if ages[newly_exposed]==1 else p_A_lowrisk
            nextstage[newly_exposed],time_to_nextstage[newly_exposed] = get_random_course_of_infection_E_to_A_or_I(p_A,params_EA,params_EI)
            dict_transmission_probs.update({newly_exposed:get_transmission_probs(time_to_nextstage[newly_exposed],b_I if nextstage[newly_exposed]=='I' else b_A,activity_highrisk if ages[newly_exposed] else 1,activity_distancers if distancing[newly_exposed] else 1)})

        if len_H_highrisk+len_H_lowrisk>max_len_H:
            max_len_H = len_H_highrisk+len_H_lowrisk
            
        if len_I_highrisk+len_I_lowrisk>max_len_I:
            max_len_I = len_I_highrisk+len_I_lowrisk
            
        len_not_tested = len_I_not_tested_highrisk+len_I_not_tested_lowrisk+len_H_not_tested_highrisk+len_H_not_tested_lowrisk
        if len_not_tested>max_len_not_tested:
            max_len_not_tested = len_not_tested

        if OUTPUT:
            res.append([len(S),len(VS),len(VI),len(E),len(A),len(I),len(H),len(R),len(RA),len(RP),len(D)])
           
        if I==[] and E==[] and H==[] and A==[]:
            if OUTPUT:
                print('stopped after %i iterations because there were no more exposed or infected or hospitalized' % t)
            break
        elif OUTBREAK==False and sum(counter_secondary_infections)/N>0.01:
            OUTBREAK=True
            if ONLY_LOOK_FOR_OUTBREAK:
                break  
    if OUTPUT:
        return res,seed,counter_secondary_infections,time_infected,disease_generation_time,total_edge_weight,max_len_H,max_len_I,max_len_not_tested
    else:
        return len(S),len(VS),len(VI),len(R),len(RA),len(RP),len(D),OUTBREAK,actual_clustering_vacc,actual_clustering_dist,actual_clustering_risk,actual_corr_vd,actual_corr_vr,actual_corr_dr,seed,counter_secondary_infections[initial_exposed],np.nanmean(disease_generation_time),np.nanmean(time_infected),infections_caused_byE,infections_caused_byA,infections_caused_byI,total_edge_weight,max_len_H,max_len_I,max_len_not_tested


#fixed parameters
#N,k,p_edgechange,p_highrisk (and nsim) as command line arguments

network_generating_function = nx.watts_strogatz_graph #can theoretically also use nx.newman_watts_strogatz_graph but then there is not the average same number of private and public interactions
params_EA = 5 #mean of Poisson RV
params_EI = 5 #mean of Poisson RV
params_IH = 8 #mean of Poisson RV
params_AR = 20 #mean of Poisson RV
params_IR = 20 #mean of Poisson RV
params_HR = 12 #mean of Poisson RV
p_IH = 0.07 #P(H|I) = 1 - P(R|I), probability of eventually moving from symptomatic class (I) to hospitalized class (H)
overall_death_rate_covid19 = 0.01
activity_reduction_H = 1
hospital_beds_per_person = 3/1000 * 2
care_decline_exponent = 0.5
private_highrisk_lowrisk_activity = 1#1=no change in behavior, 0=all contacts stopped; don't analyze distancing between highrisk/lowrisk , rather focus on distancing of highrisk 

#randomly sampled from parameter space
initial_seed = np.random.randint(1,2**32 - 1)


#to speed things up, interpolate the per day death rate once and fit every time using this interpolator, works as long as the parameters params_HR are constant, which will be checked
interpolator_per_day_death_rate = [params_HR,get_optimal_per_day_death_rate(params_HR,0.01,True)]

neverinfected_counts = []
death_counts = []
outbreaks = []
actual_clustering_vaccs = []
actual_clustering_dists = []
actual_clustering_risks = []
actual_corr_vds = []
actual_corr_vrs = []
actual_corr_drs = []
R0s = []
mean_generation_times = []
mean_time_infections = []
infections_caused_byEs = []
infections_caused_byAs = []
infections_caused_byIs = []
total_initial_edge_weights = []
max_len_Is = []
max_len_Hs = []
max_len_not_testeds = []

args = []
for i in np.arange(nsim):
    #randomly sampled from parameter space
    b_I = np.random.uniform(0.05,0.4)#(0.05,0.2)
    b_A_over_b_I = np.random.uniform(0,1)
    
    private_activity_SEA = 1/np.sqrt(2)#np.random.uniform(0,1)**(1/2)#change of private behavior for ppl in SEA: 1=no change in behavior, 0=all contacts stopped 
    public_activity_SEA = 0#private_activity_SEA#np.random.uniform(0,1)**(1/2)#change of public behavior for ppl in SEA: 1=no change in behavior, 0=all contacts stopped 
    activity_reduction_I = np.random.uniform(0,1) #change of public behavior for ppl in I: 0=no change in behavior, 1=all contacts stopped 
    activity_reduction_P = np.random.uniform(0.8,1)#change of public behavior for ppl in P (tested positive): 0=no change in behavior, 1=all contacts stopped 
    activity_reduction_highrisk = 0#1-np.random.uniform(0,1)**(1/2)#change of public behavior for highrisk ppl: 0=no change in behavior, 1=all contacts stopped 
    
    p_A = np.random.uniform(0.05,0.5)
    p_A_lowrisk_over_p_A_highrisk = np.random.uniform(1,5)
    p_IH_highrisk_over_p_IH_lowrisk = np.random.uniform(4,10)
    p_HD_highrisk_over_p_HD_lowrisk = np.random.uniform(4,10)
    
    proportion_vaccinated = 2/3#np.random.uniform(0.5,0.8)
    proportion_distancers = 2/3#np.random.uniform(0.5,0.8)
    activity_reduction_distancers = 0.5#1-1/np.sqrt(2)#np.random.uniform(0,1)
    private_activity_V = private_activity_SEA#np.random.uniform(private_activity_SEA,1)#private_activity_SEA##
    public_activity_V = 0#private_activity_V
    expected_clustering_v = proportion_vaccinated**2+(1-proportion_vaccinated)**2
    clustering_vaccine = random.choice([expected_clustering_v,expected_clustering_v+0.5*(1-expected_clustering_v)])
    #clustering_vaccine = np.random.uniform(expected_clustering_v,expected_clustering_v+0.5*(1-expected_clustering_v))
    
    expected_clustering_d = proportion_distancers**2+(1-proportion_distancers)**2
    clustering_distancing = random.choice([expected_clustering_d,expected_clustering_d+0.5*(1-expected_clustering_d)])
    #clustering_distancing = np.random.uniform(expected_clustering_d,expected_clustering_d+0.5*(1-expected_clustering_d))
    
    expected_clustering_r = p_highrisk**2+(1-p_highrisk)**2
    clustering_highrisk = random.choice([expected_clustering_r,expected_clustering_r+0.5*(1-expected_clustering_r)])
    #clustering_highrisk = np.random.uniform(expected_clustering_r,expected_clustering_r+0.5*(1-expected_clustering_r))

    #vaccine_efficacy	 = int(random.random()*6)/5#np.random.uniform(0,1)
    vaccine_efficacy = int(random.random()*6)/5#np.random.uniform(0,1)#np.random.uniform(0,1)
    
    #corr_vd = np.random.uniform(-0.15,0.15)
    #corr_vr = np.random.uniform(-0.15,0.15)
    #corr_dr = np.random.uniform(-0.15,0.15)
    
    corr_vd = random.choice([-0.15,0,0.15])
    corr_vr = random.choice([-0.15,0,0.15])
    corr_dr = random.choice([-0.15,0,0.15])
    corr_vdr = 0


    triaging_policy = 'FBLS'#np.random.choice(['FCFS','same','FBLS','FBrnd'])
    max_number_of_tests_available_per_day = 0
    if N==1000:
        max_number_of_tests_available_per_day = np.random.choice([0,1,2,3,4,5,6,7,8,9,10,20,40])
    else:
        max_number_of_tests_available_per_day = np.random.choice([0,5,10,20,30,40,50,60,70,80,100,200,400])
    testing_policy_decision_major = 'Y>O'#np.random.choice(['O>Y','Y>O'])
    testing_policy_decision_minor = 'LIFT'#np.random.choice(['random','FIFT','LIFT'])
    testing_delay = np.random.randint(0,8)
    
    #derived/fitted parameters    
    b_A = b_A_over_b_I*b_I
    b_H = b_I
    p_IH_lowrisk_over_p_IH_highrisk = 1/p_IH_highrisk_over_p_IH_lowrisk
    p_HD_lowrisk_over_p_HD_highrisk = 1/p_HD_highrisk_over_p_HD_lowrisk
    probability_of_close_contact_with_random_person_in_public = k/(N-1)
    private_activity_I = private_activity_SEA*(1-activity_reduction_I)#change of private behavior for ppl in I: 1=no change in behavior, 0=all contacts stopped 
    public_activity_I = public_activity_SEA*(1-activity_reduction_I)#change of public behavior for ppl in I: 1=no change in behavior, 0=all contacts stopped 
    private_activity_H = private_activity_SEA*(1-activity_reduction_H)#change of private behavior for ppl in H: 1=no change in behavior, 0=all contacts stopped 
    public_activity_H = public_activity_SEA*(1-activity_reduction_H)#change of public behavior for ppl in H: 1=no change in behavior, 0=all contacts stopped 
    activity_P = 1-activity_reduction_P#change of public behavior for ppl in P (tested positive): 1=no change in behavior, 0=all contacts stopped 
    activity_highrisk = 1-activity_reduction_highrisk#change of public behavior for highrisk ppl: 1=no change in behavior, 0=all contacts stopped 
    activity_distancers = 1-activity_reduction_distancers
    while True: #for whatever reason  clustering_v2 sometimes (rarely) causes an index error
        try:
            len_S,len_VS,len_VI,_,_,_,death_count,outbreak,actual_clustering_vacc,actual_clustering_dist,actual_clustering_risk,actual_corr_vd,actual_corr_vr,actual_corr_dr,seed,R0,mean_generation_time,mean_time_infection,infections_caused_byE,infections_caused_byA,infections_caused_byI,total_initial_edge_weight,max_len_H,max_len_I,max_len_not_tested = model(N,k,p_edgechange,network_generating_function,p_highrisk,T_max,b_A,b_I,b_H,params_EA,params_EI,params_IH,params_AR,params_IR,params_HR,p_A,p_A_lowrisk_over_p_A_highrisk,p_IH,p_IH_lowrisk_over_p_IH_highrisk,overall_death_rate_covid19,p_HD_lowrisk_over_p_HD_highrisk,probability_of_close_contact_with_random_person_in_public,private_activity_SEA,private_activity_I,private_activity_H,private_highrisk_lowrisk_activity,public_activity_SEA,activity_highrisk,activity_distancers,public_activity_I,public_activity_H,hospital_beds_per_person,care_decline_exponent,triaging_policy,max_number_of_tests_available_per_day,activity_P,testing_policy_decision_major,testing_policy_decision_minor,testing_delay,proportion_vaccinated,vaccine_efficacy,private_activity_V,public_activity_V,proportion_distancers,clustering_vaccine,clustering_distancing,clustering_highrisk,corr_vd,corr_vr,corr_dr,corr_vdr,seed=None,OUTPUT=False,ESTIMATE_R0=True,interpolator_per_day_death_rate=interpolator_per_day_death_rate)
            break
        except IndexError:
            pass
    neverinfected_count = len_S+len_VS+len_VI
    neverinfected_counts.append(neverinfected_count)
    death_counts.append(death_count)
    outbreaks.append(outbreak)
    actual_clustering_vaccs.append(actual_clustering_vacc)
    actual_clustering_dists.append(actual_clustering_dist)
    actual_clustering_risks.append(actual_clustering_risk)
    actual_corr_vds.append(actual_corr_vd)
    actual_corr_vrs.append(actual_corr_vr)
    actual_corr_drs.append(actual_corr_dr)
    R0s.append(R0)
    mean_generation_times.append(mean_generation_time)
    mean_time_infections.append(mean_time_infection)
    infections_caused_byEs.append(infections_caused_byE)
    infections_caused_byAs.append(infections_caused_byA)
    infections_caused_byIs.append(infections_caused_byI)
    total_initial_edge_weights.append(total_initial_edge_weight)
    max_len_Hs.append(max_len_H)
    max_len_Is.append(max_len_I)
    max_len_not_testeds.append(max_len_not_tested)
    
    args.append([N,k,p_edgechange,network_generating_function,p_highrisk,T_max,b_I,b_A_over_b_I,params_EA,params_EI,params_IH,params_AR,params_IR,params_HR,p_A,p_A_lowrisk_over_p_A_highrisk,p_IH,p_IH_lowrisk_over_p_IH_highrisk,overall_death_rate_covid19,p_HD_lowrisk_over_p_HD_highrisk,probability_of_close_contact_with_random_person_in_public,private_activity_SEA,public_activity_SEA,activity_reduction_I,activity_reduction_H,activity_reduction_highrisk,activity_reduction_distancers,private_highrisk_lowrisk_activity,hospital_beds_per_person,care_decline_exponent,triaging_policy,max_number_of_tests_available_per_day,activity_reduction_P,testing_policy_decision_major,testing_policy_decision_minor,testing_delay,proportion_vaccinated,vaccine_efficacy,private_activity_V,public_activity_V,proportion_distancers,clustering_vaccine,clustering_distancing,clustering_highrisk,corr_vd,corr_vr,corr_dr,corr_vdr,seed])
args = np.array(args)
#to get args_names, run this ','.join(["'"+el+"'" for el in '''N,k,p_edgechange,network_generating_function,p_highrisk,T_max,b_I,b_A_over_b_I,params_EA,params_EI,params_IH,params_AR,params_IR,params_HR,p_A,p_A_lowrisk_over_p_A_highrisk,p_IH,p_IH_lowrisk_over_p_IH_highrisk,overall_death_rate_covid19,p_HD_lowrisk_over_p_HD_highrisk,probability_of_close_contact_with_random_person_in_public,private_activity_SEA,public_activity_SEA,activity_reduction_I,activity_reduction_H,activity_reduction_highrisk,activity_reduction_distancers,private_highrisk_lowrisk_activity,hospital_beds_per_person,care_decline_exponent,triaging_policy,max_number_of_tests_available_per_day,activity_reduction_P,testing_policy_decision_major,testing_policy_decision_minor,testing_delay,proportion_vaccinated,vaccine_efficacy,private_activity_V,public_activity_V,proportion_distancers,clustering_vaccine,clustering_distancing,clustering_highrisk,corr_vd,corr_vr,corr_dr,corr_vdr,initial_seed'''.split(',')])
args_names = ['N','k','p_edgechange','network_generating_function','p_highrisk','T_max','b_I','b_A_over_b_I','params_EA','params_EI','params_IH','params_AR','params_IR','params_HR','p_A','p_A_lowrisk_over_p_A_highrisk','p_IH','p_IH_lowrisk_over_p_IH_highrisk','overall_death_rate_covid19','p_HD_lowrisk_over_p_HD_highrisk','probability_of_close_contact_with_random_person_in_public','private_activity_SEA','public_activity_SEA','activity_reduction_I','activity_reduction_H','activity_reduction_highrisk','activity_reduction_distancers','private_highrisk_lowrisk_activity','hospital_beds_per_person','care_decline_exponent','triaging_policy','max_number_of_tests_available_per_day','activity_reduction_P','testing_policy_decision_major','testing_policy_decision_minor','testing_delay','proportion_vaccinated','vaccine_efficacy','private_activity_V','public_activity_V','proportion_distancers','clustering_vaccine','clustering_distancing','clustering_highrisk','corr_vd','corr_vr','corr_dr','corr_vdr','initial_seed']

f = open(output_folder+'output_model%s_nsim%i_N%i_k%i_seed%i_SLURM_ID%i.txt' % (version,nsim,N,k,initial_seed,SLURM_ID) ,'w')
f.write('filename\t'+filename+'\n')
f.write('SLURM_ID\t'+str(SLURM_ID)+'\n')
for ii in range(len(args_names)):#enumerate(zip(args,args_names)):
    if args_names[ii] in ['network_generating_function'] or len(set(args[:,ii])) == 1:
        f.write(args_names[ii]+'\t'+str(args[0,ii])+'\n')    
    else:
        f.write(args_names[ii]+'\t'+'\t'.join(list(map(str,[el if type(el)!=float else round(el,9) for el in args[:,ii]])))+'\n')

f.write('neverinfected_counts\t'+'\t'.join(list(map(str,neverinfected_counts)))+'\n')
f.write('death_counts\t'+'\t'.join(list(map(str,death_counts)))+'\n')
f.write('outbreaks\t'+'\t'.join(list(map(str,outbreaks)))+'\n')
f.write('actual_clustering_vaccination\t'+'\t'.join(list(map(str,actual_clustering_vaccs)))+'\n')
f.write('actual_clustering_distancing\t'+'\t'.join(list(map(str,actual_clustering_dists)))+'\n')
f.write('actual_clustering_highrisk\t'+'\t'.join(list(map(str,actual_clustering_risks)))+'\n')
f.write('actual_correlation_vacc_dist\t'+'\t'.join(list(map(str,actual_corr_vds)))+'\n')
f.write('actual_correlation_vacc_risk\t'+'\t'.join(list(map(str,actual_corr_vrs)))+'\n')
f.write('actual_correlation_dist_risk\t'+'\t'.join(list(map(str,actual_corr_drs)))+'\n')
f.write('R0s\t'+'\t'.join(list(map(str,[round(el,3) for el in R0s])))+'\n')
f.write('mean_generation_times\t'+'\t'.join(list(map(str,[round(el,3) for el in mean_generation_times])))+'\n')
f.write('mean_time_infections\t'+'\t'.join(list(map(str,[round(el,3) for el in mean_time_infections])))+'\n')
f.write('infections_caused_byEs\t'+'\t'.join(list(map(str,[round(el,3) for el in infections_caused_byEs])))+'\n')
f.write('infections_caused_byAs\t'+'\t'.join(list(map(str,[round(el,3) for el in infections_caused_byAs])))+'\n')
f.write('infections_caused_byIs\t'+'\t'.join(list(map(str,[round(el,3) for el in infections_caused_byIs])))+'\n')
f.write('total_initial_edge_weights\t'+'\t'.join(list(map(str,[round(el,6) for el in total_initial_edge_weights])))+'\n')
f.write('max_len_Hs\t'+'\t'.join(list(map(str,[round(el,3) for el in max_len_Hs])))+'\n')
f.write('max_len_Is\t'+'\t'.join(list(map(str,[round(el,3) for el in max_len_Is])))+'\n')
f.write('max_len_not_testeds\t'+'\t'.join(list(map(str,[round(el,3) for el in max_len_not_testeds])))+'\n')
f.close()
