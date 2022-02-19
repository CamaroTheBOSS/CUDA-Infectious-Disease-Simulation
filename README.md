# CUDA-Infectious-Disease-Simulation

 <font size="20"> 1.0 Introduction </font>

  Infectious disease's spreading simulation is leading topic today. In case of more complex models is possible to forecast diesease's
  spreading that could help in preparing for next epidemic waves. Creating model on GPU could make the simulation more real,
  because it can simulate bigger society (more agents at the same time)
  
  
2.0 Program description

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
  
  
3.0 Infection system
  
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
  
  
 4.0 Simulation's parameters description
 
  It is possible to change simulation parameters in main file parallel.cu in SetSimParameters() function:
  
  General simulation's parameters:\
  nAgentsx - tells us about numebr of simulated agents (on GPU it has to be 2^n because of the BitonicShuffler),\
  simTimex - tells us about number of simulated days,\
  vaccinTimex - tells us about vaccination's effectivness (time of protection (fresh vaccinations have better protection than
  elder vaccinations)),\
  nJourneyx - tells us about number of visited places each day,\
  nInfectedAgentsProcentx - tells us about % of agents who are infected at the start of the simulation,\
  nGroupsx - tells us about number of micro groups, the agents are part of (on GPU it has to be 2^n because of the
  BitonicShuffler).
  
  Agents borders:
  
  maxDeathProbbx - tells us about max probability of death when being infected which agent can assume during initialization,\
  maxInfectProbx - tells us about max probability of infect someone other when being infected which agent can assume during
  initialization,\
  maxAgentRessistancex - tells us about max ressistance which agent can assume during initialization,\
  maxSwapProbx - tells us about max probability of wearing mask by agents after each day,\
  maskEffectivnessx - effectivness of being masked,\
  maxVaccinationProbx - tells us about max probability of being vaccinated each day (when is not vaccinated) which agent can
  assume during initialization.
  
  Disease's parameters:
  
  mutuationProbx - change for disease's mutuation,\
  mutuationIntensityx - intensity of each mutuation,\
  contagiousnesx - contagiousness of disease,\
  durationx - average duration of being infected,\
  mutationTime - time (in days) during which agents might change their state from convalescent to healthy (initializate after
  each mutuation),\
  convalescentToHealthyProb - probability of changing state from convalescent to healthy in mutuationTime after disease's
  mutuation.
  
  Visited places parameters:
  
  maxContactFactorx - tells us about max contact factor that place can assume during initialization,\
  nPlacesCPUx (only for CPU) - tells us about number of simulated places agents can visit.
