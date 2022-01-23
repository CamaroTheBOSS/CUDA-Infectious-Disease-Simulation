// Simulation parameters ------------------------------------------
#define nAgents 1000           //27,3kB for each 1k agents
#define simTime 10*365		  //4,4kB for each year (simTime = 365 is one year simulation)
#define vaccinTime 365       //time of protection caused vaccination
#define nJourney 3          //number of journeys each day for each agent
#define nInfectedAgents 10 //defines how many agents are infected at the start

// Agents boundaries ----------------------------------------------
#define maxGetInfectedValue 0.1
#define maxDeathProb 0.05
#define maxInfectProb 0.1
#define maxExtravertizmParameter 1

// Disease parameters ----------------------------------------------
#define mutuationProb 0.01      //chance for disease mutuation each day
#define mutuationIntensity 0.1 //defines each mutuation's influence
#define Dcontagiousness 0.1   //probability for getting infected by disease
#define Dduration 14         //avarage disease's duration

// Places parameters -----------------------------------------------
#define avrCapacity 500        //avarage capacity of places
#define standardDeviation 470 //max deviation from avrCapacity, has to be smaller than avrCapacity
#define nPlaces 5			 //7,8kB for each 1k places
