#include "parameters.h"
#include "randomf.h"

struct Agent
{
	bool masked = false;           // masked agents has better ressistance
	
	float deathProb = 0;        // probability for death caused by infection
	float infectProb = 0;      // probability for infecting someone other
	float ressistance = 0;    // disease ressistance
	float swapMaskProb = 0;  // probability for wear/unwear the mask for the next day
	float vaccinWill = 0;   // will for beeing vaccinated

	unsigned __int16 vacRessist = 0;    // infection ressistance caused vaccination (for given number of days)
	unsigned __int8 state = 0;         // 0 = healthy ; 1 = infected ;  2 = convalescent; 3 = dead
	unsigned __int8 sickDaysLeft = 0; // how many days to infection's end
	unsigned __int8 randN = 0;
};