#include "parameters.h"
#include "randomf.h"

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
	int placeIdx = 1; //place index the Agent is in

	Agent::Agent(void)
	{
		getInfectedProb = floatRand(0, maxGetInfectedValue);
		deathProb = floatRand(0, maxDeathProb);
		infectProb = floatRand(0, maxInfectProb);
		extrovertizm = floatRand(0, maxExtravertizmParameter);
	}

	float getParameter(int i)
	{
		if (i == 0)
		{
			return getInfectedProb;
		}
		else
		{
			return deathProb;
		}
	}

};