// Simulation parameters ------------------------------------------
#define nAgents 65000           //27,3kB for each 1k agents
#define simTime 1*365		  //4,4kB for each year (simTime = 365 is one year simulation)
#define vaccinTime 365       //time of protection caused vaccination
#define nJourney 3          //number of journeys each day for each agent
#define nInfectedAgents 0.01 //defines infected agents procent at the start [%]

// Agents boundaries ----------------------------------------------
#define maxDeathProb 0.01             // max probability for death when beeing infected
#define maxInfectProb 0.05           // max probability for infecting someone other (decreased by wearing a mask)
#define maxRessistanceParameter 0.7 // max infection ressistance
#define maxSwapMaskProb 0.1        // max probability for wear/unwear the mask for the next day
#define maxVaccinProb 0.2         // max will for beeing vaccinated

// Disease parameters ----------------------------------------------
#define mutuationProb 0.01      //chance for disease mutuation each day
#define mutuationIntensity 0.1 //defines each mutuation's influence
#define Dcontagiousness 0.05   //probability for getting infected by disease
#define Dduration 14         //avarage disease's duration

// Places parameters -----------------------------------------------
#define maxExtavertizm 0.1
