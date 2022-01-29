struct SimulationParameters
{
	// Simulation parameters ------------------------------------------
	unsigned int nAgentsx = 1024;
	unsigned int simTimex = 365;
	unsigned int vaccinTimex = 365;
	unsigned __int8 nJourneyx = 3;
	float nInfectedAgentsProcentx = 0.01;
	float maskEffectivnessx = 0.5;

	// Agents boundaries ----------------------------------------------
	float maxDeathProbbx = 0.01;
	float maxInfectProbx = 0.05;
	float maxAgentRessistancex = 0.7;
	float maxMaskSwapProbx = 0.1;
	float maxVaccinationProbx = 0.2;

	// Disease parameters ----------------------------------------------
	float mutuationProbx = 0.005;
	float mutuationIntensityx = 0.1;
	float contagiousnesx = 0.05;
	int durationx = 14;

	// Places parameters -----------------------------------------------
	float maxContactFactorx = 0.1;
	int nPlacesCPUx = 2;
};