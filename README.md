# CUDA-Infectious-Disease-Simulation

1. Introduction
  Infectious disease's spreading simulation is leading topic today. In case of more complex models is possible to forecast diesease's
  spreading that could help in preparing for next epidemic waves. Creating model on GPU could make the simulation more real,
  because it can simulate bigger society (more agents at the same time)
  
2. Program description
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
  interior loop where agents are choosing specific place 
