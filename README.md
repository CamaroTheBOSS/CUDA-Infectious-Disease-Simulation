# CUDA-Infectious-Disease-Simulation

## 1.0 Introduction

  Infectious disease's spreading simulation is leading topic today. In case of more complex models is possible to forecast diesease's
  spreading that could help in preparing for next epidemic waves. Creating model on GPU could make the simulation more real,
  because it can simulate bigger society (more agents at the same time)
  
  
## 2.0 Program description

  Program structure is quite simple. Class SimulationParameters gives most important simulation's parameters like number
  of simulated agents (agents are contained in dynamic table) or number of simulated days. Each agent is a structure having their
  own features like probability of infect someone other or probability of death when being infected.
  
  Places visited by agents have their own features like contactFactor, which tells us whether given place is more contactful or not. 
  For example football pitch will be more contactful than theatre. Programs on GPU and CPU are a bit diffrent. On GPU for each 
  block of threats we have 2 given places and random border, which tells us about a capacity of each place. For example in GPU block 
  with 1024 threads and border with value 400 first 400 threads are in 1 place and 624 next threads are in 2 place. It means that 
  first 400 threads are interacting with each other and next 624 threads are interacting with each other. So we have 2 times more 
  places than GPU blocks. On CPU number of diffrent places is setted in simulation's parameters, so it is possible to do simulation 
  with n agents and n places. In this situation nobody should infect another agent. On CPU each place has the same capacity 
  (number of simulated agents divided by number of places). Common thing for both programs is that the places are always full.
  
  The way the agents are choosing place is randomized. Before choosing places, the dynamic table with agents are shuffled. In
  interior loop where agents are choosing specific place they are shuffled in micro scale (agents create some groups and they are
  shuffled in range of the specific group). Every week agents are shuffled in macro scale (agents are shuffled between groups).
  In reality people usually meet with the same people so this is great approximation. On GPU shuffling is working on reworked
  Bitonic Sort Algorithm, which is sorting agents according to assigned to them random values. On CPU micro shuffling is working
  on reworked Quick Sort Algorithm, which is sorting agents according to assigned to them random values as well, but macro shuffling
  is working based on random draw using a set. This algorithm is faster than Quick Sort, but hard to implement for multiple groups
  without knowledge about number of groups.
  
  After visiting n places agents are updated. Some agents wearing masks, some of them are vaccinated. It is tested whether the agent
  dies on specific day and every interior agents variables are updated (time to recovery-- etc.)
  
  Whole program is working based on two nested loops. The exterior loop tells us about day, but interior loop tells us about
  number of place to visit in specific day. Program output is table with number of healthy, infected, convalescent and dead
  agents. Program on GPU is working in parallel of course. Program on CPU is supported with parallel computations, because of the
  OpenMP library. Finally the results are saved in file and we can display them in simple python script using matplotlib
  and pandas libraries.
  
  
## 3.0 Infection system
  
  Each day agents are visiting n places (depends on simulation parameters). In each visited place they interact with each other.
  Agents from place x have interactions with another agents from place x etc. Healthy agents in place x could by infected
  by infected agents in place x. Agent infection probability depends on disease's contagiousness, number of infected agents in
  specific place, parameters which tells us about a probability of infect someone other by the specific agent, specific place's
  contact factor, whether infected agents wear masks, personal ressistance of each agent and whether agent is vaccinated.
  If infection test is positive agent changes his status from healthy to infected and is assigned to him time (in days) to 
  recovery. After this time agents changes his status from infected to convalescent. Moreover each day for infected agents is
  tested death probability. If is positive agent changes his status from infected to dead.
  
  Convacelscent agents cannot get infected, but in program disease's mutuations are simulated. After mutuation not only the 
  disease's contagiousness is changing, but more important, some of the convalescent agents might change their status to healthy
  and they have another change for getting infected one more time.
  
  
 ## 4.0 Simulation's parameters description
 
  It is possible to change simulation parameters in main file parallel.cu in SetSimParameters() function:
  
  #### General simulation's parameters:
  nAgentsx - tells us about numebr of simulated agents (on GPU it has to be 2^n because of the BitonicShuffler),\
  simTimex - tells us about number of simulated days,\
  vaccinTimex - tells us about vaccination's effectivness (time of protection (fresh vaccinations have better protection than
  elder vaccinations)),\
  nJourneyx - tells us about number of visited places each day,\
  nInfectedAgentsProcentx - tells us about % of agents who are infected at the start of the simulation,\
  nGroupsx - tells us about number of micro groups, the agents are part of (on GPU it has to be 2^n because of the
  BitonicShuffler).
  
  #### Agents borders:
  
  maxDeathProbbx - tells us about max probability of death when being infected which agent can assume during initialization,\
  maxInfectProbx - tells us about max probability of infect someone other when being infected which agent can assume during
  initialization,\
  maxAgentRessistancex - tells us about max ressistance which agent can assume during initialization,\
  maxSwapProbx - tells us about max probability of wearing mask by agents after each day,\
  maskEfficiency - effectivness of being masked,\
  maxVaccinationProbx - tells us about max probability of being vaccinated each day (when is not vaccinated) which agent can
  assume during initialization.
  
  #### Disease's parameters:
  
  mutuationProbx - change for disease's mutuation,\
  mutuationIntensityx - intensity of each mutuation,\
  contagiousnesx - contagiousness of disease,\
  durationx - average duration of being infected,\
  mutationTime - time (in days) during which agents might change their state from convalescent to healthy (initializate after
  each mutuation),\
  convalescentToHealthyProb - probability of changing state from convalescent to healthy in mutuationTime after disease's
  mutuation.
  
  #### Visited places parameters:
  
  maxContactFactorx - tells us about max contact factor that place can assume during initialization,\
  nPlacesCPUx (only for CPU) - tells us about number of simulated places agents can visit.

## 5.0 Testing platform:

  CPU - Intel Core i5 10400F\
  GPU -NVIDIA Geforce RTX3060 (Ampere generation)
  
## 6.0 Tests which compare GPU and CPU programs:

  At the start the CPU simulation was tested. All the charts present dependence nAgents in specific state(specific day in 
  simulation):
  
  ![image](https://user-images.githubusercontent.com/67116759/154803283-c2b3a227-e52b-4052-b94b-b27bec13cd54.png)\
  *Draw 1: Tested simulation's parameters*

  Results are presented on chart:
  
  ![image](https://user-images.githubusercontent.com/67116759/154803302-e4275727-7d33-4f64-942d-61812b93ad72.png)\
  *Draw 2: Results of simulation for 1024 agents on CPU*
  
  Simulation time oscilate between 30 and 50ms for each day. It means that this simulation took 40sec. For comparation
  the simulation below is performed on GPU with the same parameters:
  
  ![image](https://user-images.githubusercontent.com/67116759/154803334-6591cf67-a9a4-433c-bc94-003923306786.png)\
  *Draw 3: Results of simulation for 1024 agents on GPU*
  
  It is worth to say that results could be diffrent because of the random character of the simulation. Time of this simulation
  oscilate between 13 and 14ms for each day. It means that this simulation took 14.5sec. It is significant acceleration, 
  but the number of agents are too little for showing the whole advantage of using GPU for this simulation. Below I have shown
  results that tells us about border of both solutions:
  
  ![image](https://user-images.githubusercontent.com/67116759/154803347-4c7b5fc1-4fb2-4ece-8ce7-da82ab94f937.png)\
  *Draw 4: Results of simulation for 4096 agents on CPU*
  
  Simulation time oscilate between 580ms and 800ms, so whole simulation took 11.5min. It is too long time relative to number
  of simulated agents. Below I have shown simulation on GPU:
  
  ![image](https://user-images.githubusercontent.com/67116759/154803360-e8825900-b5d7-43f7-af8b-d970378b4425.png)\
  *Draw 5: Results of simulation for 1048576 agents on GPU*
  
  Simulation time oscilate between 700 and 800ms, so for 1095 simulated days it took 13.5min for whole simulation. 2 min longer
  than on CPU, but with 250 times more simulated agents. The conclusion is that in case of simulating systems with a lot of 
  individual objects (agents etc.) it is great to use GPU for this purpose. It can give significant acceleration relative to CPU,
  so the simulatuon can give better results and be more predictive, proffesional, real.
  
  ## 7.0 Tests with diffrent datasets on GPU:
  
  Below I have shown simulation for maxMaskSwapProbx = 0, maxVaccinationProbx = 0 and for maxMaskSwapProbx = 1, 
  maxVaccinationProbx = 1.
  
  ![image](https://user-images.githubusercontent.com/67116759/154803369-d1e9a58e-e078-4120-9e5f-61af3b6d8d1d.png)\
  *Draw 6: Results of simulation for maxMaskSwapProbx=0 and maxVaccinationProbx=0 for 131072 agents on GPU*
  
  ![image](https://user-images.githubusercontent.com/67116759/154803375-ad987c08-2f5f-47f0-a1da-de1d7fb1f669.png)\
  *Draw 7: Results of simulation for maxMaskSwapProbx=1 and maxVaccinationProbx=1 for 131072 agents on GPU*
  
  We can see that in simulation without masks and vaccination first wave of epidemic is longer, harder to suppress. In the
  second case, with big % of masked and vaccinated peoople, next waves disappear faster. Below I have shown similar simulation
  with perfect effectivness of masks:
  
  ![image](https://user-images.githubusercontent.com/67116759/154803383-aecb14fc-821d-4cd5-9c53-be95e02ddfb5.png)\
  *Draw 8: Results of simulation for maxMaskSwapProbx=1, maxVaccinationProbx=1 and maskEfficiency=1 for 131072 agents on GPU*
  
  If masks were perfect solution (100% effectivness) and for big % of masked agents, after first wave the long stagnation would
  occur and disease would be defeated. Below I have shown simulation of society with decreased ressistance (from 0.5 to 0.3):
  
  ![image](https://user-images.githubusercontent.com/67116759/154803392-dfe0ab1f-035a-4c8a-92cc-2a2997854850.png)\
  *Draw 9: Results of simulation for maxMaskSwapProbx=1, maxVaccinationProbx=1 and maxRessistance=0.3 for 131072 agents on GPU*
  
  We can see that society with decreased ressistance is more susceptible for being infected. Next waves are stronger, number
  of death agents is bigger. The conclusion is that it is important to take care about ourself, about physical health and 
  mental health, diet and higiene. By doing these things we can become more ressistant for disease. Below I have shown results
  of simulation in case of two times longer time of disease's duration when being infected:
  
  ![image](https://user-images.githubusercontent.com/67116759/154803396-bf682082-2438-48c2-90e6-9bd5a0f6e5cf.png)\
  *Draw 10: Results of simulation for maxMaskSwapProbx=1, maxVaccinationProbx=1, durationx=18 and mutuationTime=8 for 131072 
  agents on GPU*
  
  Making disease longer deepened problem of next waves significantly. The conclusion is that in case of virus detection in
  ourselfs it is important to take care about ourself, relax and go for quarantine.
