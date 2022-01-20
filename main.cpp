#include <iostream>

// Simulation parameters ------------------------------------------
#define nAgents 1000
#define simTime 10*365
#define vaccinTime 365 //time of protection caused vaccination
#define nJourney 3 //number of journeys each day for each agent
#define nInfectedAgents 10 //defines how many agents are infected at the start

// Agents boundaries ----------------------------------------------
#define maxGetInfectedValue 0.1
#define maxDeathProb 0.05
#define maxInfectProb 0.1
#define maxExtravertizmParameter 1

// Disease parameters ----------------------------------------------
#define mutuationProb 0.01 //chance for disease mutuation each day
#define mutuationIntensity 0.1 //defines each mutuation's influence
#define Dcontagiousness 0.1 //probability for getting infected by disease
#define Dduration 14 //avarage disease's duration

float floatRand(float fMin, float fMax)
{
	float f = (float)rand() / RAND_MAX;
	return fMin + f * (fMax - fMin);
}

class Place
{
private:
	int cap = 0; //max number of people which can be in place
	int size = 0; //number of people who currently are in given place
	int contactFactor = 0; //increase risk of beeing infected (0, 1) (bigger contactFactor means that place gives better enviroment for beeing infected e.g. a lot of people close to each other - festival or concert)

public:
	Place::Place(int capacity, int contactFac)
	{
		cap = capacity;
		contactFactor = contactFac;
	}
};

class Disease
{
private:

	float contagiousness = 0; //probability for getting infected by disease
	float duration = 0; // of the disease

public:

	Disease::Disease(void)
	{
		contagiousness = Dcontagiousness;
		duration = Dduration;
	}
};


class Agent
{
private:

	bool masked = false; //masked agents has better ressistance

	float getInfectedProb = 0; //probability for getting infected
	float deathProb = 0; //probability for death caused by infection
	float infectProb = 0; // probability for infecting someone other
	float extrovertizm = 0; // 0 means that agent go to places with small number of people, 1 means that agent go to places with a lot of people, 0.5 means that agent go everywhere

	int vacRessist = 0; //infection ressistance caused vaccination (for given number of days)
	int state = 0; //0 = healthy ; 1 = infected ;  2 = convalescent
	int sickDaysLeft = 0; //how many days to infection's end

public:

	Agent::Agent(void)
	{
		getInfectedProb = floatRand(0, maxGetInfectedValue);
		deathProb = floatRand(0, maxDeathProb);
		infectProb = floatRand(0, maxInfectProb);
		extrovertizm = floatRand(0, maxExtravertizmParameter);
	}

	



};

int main()
{
	return 0;
}