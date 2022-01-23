#include "parameters.h"
#include "randomf.h"

struct Agent
{
	bool masked = false;           //masked agents has better ressistance

	float getInfectedProb = 0;   //probability for getting infected
	float deathProb = 0;        //probability for death caused by infection
	float infectProb = 0;      // probability for infecting someone other
	float extrovertizm = 0;   // 0 means that agent go to places with small number of people, 1 means that agent go to places with a lot of people, 0.5 means that agent go everywhere

	unsigned __int16 vacRessist = 0;     //infection ressistance caused vaccination (for given number of days)
	unsigned __int8 state = 0;         //0 = healthy ; 1 = infected ;  2 = convalescent
	unsigned __int8 sickDaysLeft = 0; //how many days to infection's end
	unsigned short int placeIdx = 1;    //place index the Agent is in
};