#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <chrono>

#include "Place.cuh"
#include "Disease.cuh"
#include "Agent.cuh"
#include "parameters.h"
#include "randomf.h"

void SumOutputs(Agent* agents, int* healthy, int* infected, int* convalescent, int* dead, int day)
{
	int i;
	int size = nAgents;
	healthy[day] = 0;
	infected[day] = 0;
	convalescent[day] = 0;
	dead[day] = 0;
#pragma omp parallel for shared(size, agents, healthy, infected, convalescent, dead, day), private(i)
	for (i = 0; i < size; i++)
	{
		if (agents[i].state == 0)
		{
#pragma omp critical
			{
				healthy[day] += 1;
			}	
		}
		else if (agents[i].state == 1)
		{
#pragma omp critical
			{
				infected[day] += 1;
			}
		}
		else if (agents[i].state == 2)
		{
#pragma omp critical
			{
				convalescent[day] += 1;
			}
		}
		else if (agents[i].state == 3)
		{
#pragma omp critical
			{
				dead[day] += 1;
			}
		}
	}
}

void UpdateAgents(Agent* agents)
{
	int i;
	int size = nAgents;
#pragma omp parallel for shared(size, agents), private(i)
	for (i = 0; i < size; i++)
	{
		if ((agents[i].sickDaysLeft != 0) && (agents[i].state != 3))
		{
			agents[i].sickDaysLeft--;
			if (agents[i].sickDaysLeft == 0)
				agents[i].state = 2; //convalescent
		}
		if (agents[i].vacRessist != 0)
		{
			agents[i].vacRessist--;
		}
	}
}

void DiseaseMutuation(Disease &disease)
{
	float x = floatRand(0, 1);
	if (x < mutuationProb)
	{
		x = floatRand(-mutuationIntensity, mutuationIntensity);
		disease.contagiousness += x * disease.contagiousness;
	}
}

void TestDeath(Agent* agents)
{
	int i;
	int size = nAgents;
	float x;
#pragma omp parallel for shared(size, agents), private(i, x)
	for (i = 0; i < size; i++)
	{
		if (agents[i].state == 1)
		{
			x = floatRand(0, 1);
			if (x < agents[i].deathProb)
			{
				agents[i].state = 3; //death
				//printf("Im dead :(\n");
			}
		}
	}	
}

void MaskingVaccinAgents(Agent* agents)
{
	int size = nAgents;
	int i;
	float x;
#pragma omp parallel for shared(size, agents), private(i, x)
	for (i = 0; i < size; i++)
	{
		x = floatRand(0, 1);
		if (x < agents[i].swapMaskProb)
		{
			if (agents[i].masked)
				agents[i].masked = false;
			else
				agents[i].masked = true;
		}

		x = floatRand(0, 1);
		if ((x < agents[i].vaccinWill) && (agents[i].vacRessist == 0))
		{
			agents[i].vacRessist == vaccinTime;
		}
	}
}

float TestAgent(Agent agent, float infProb)
{
	float x = floatRand(0, 1);

	float ressistanceComponent = agent.ressistance * infProb;
	float vacRessist = infProb * agent.vacRessist / vaccinTime;
	infProb -= ressistanceComponent + vacRessist;
	if (x > infProb)
	{
		return 0;
	}
	return 1;
}

void InfectionTest(Agent* agents, Disease disease, Place* places)
{
	int i, j;
	int size = nAgents;
	int placeCap = nAgents / nPlacesCPU;

	int firstIdx, lastIdx, placeIdx, nInfected;
	float infectionComponent, nAgentsComponent, diseaseComponent, placeComponent, infectionprob;


#pragma omp parallel for shared(size, placeCap, agents, places, disease), private(i, j, firstIdx, lastIdx, placeIdx, nInfected, infectionComponent, nAgentsComponent, diseaseComponent, infectionprob)
	for (i = 0; i < size; i++)
	{
		firstIdx = placeCap * floor((float)i / (float)placeCap);
		lastIdx = firstIdx + placeCap;
		placeIdx = floor((float)i / (float)placeCap);
		
		nInfected = 0;

		infectionComponent = 0;
		nAgentsComponent = 0;
		diseaseComponent = 0;
		placeComponent = 0;
		infectionprob = 0;
		for (j = firstIdx; j < lastIdx; j++)
		{
			if (agents[j].state == 1)
			{
				nInfected++;
				infectionComponent += agents[j].infectProb;
				
				if (agents[j].masked)
				{
					infectionComponent -= agents[j].infectProb * maskEffectivness;
				}
			}
		}
		infectionComponent /= (nInfected + 1);
		nAgentsComponent = infectionComponent * (float)nInfected / (float)placeCap;
		diseaseComponent = (float)(disease.contagiousness * nInfected) / (float)placeCap;
		placeComponent = (float)places[placeIdx].contactFactor * nInfected / (float)placeCap;
		infectionprob = infectionComponent + nAgentsComponent + diseaseComponent + placeComponent;

		if (infectionprob > 1)
		{
			infectionprob = 1;
		}
		//printf("%f, %f, %f, %f, %f\n",infectionComponent, nAgentsComponent, diseaseComponent, placeComponent, infectionprob);

		if (agents[i].state == 0)
		{
			if (TestAgent(agents[i], infectionprob))
			{
				agents[i].state = 1;
				agents[i].sickDaysLeft = Dduration;
				//printf("Im sick :(\n");
			}
		}
	}
}

void InitAgents(Agent* agents)
{
	int i;
	int size = nAgents;
#pragma omp parallel for shared(size, agents), private(i)
	for (i = 0; i < size; i++)
	{
		agents[i].deathProb = floatRand(0, maxDeathProb);
		agents[i].infectProb = floatRand(0, maxInfectProb);
		agents[i].ressistance = floatRand(0, maxRessistanceParameter);
		agents[i].swapMaskProb = floatRand(0, maxSwapMaskProb);
		agents[i].vaccinWill = floatRand(0, maxVaccinProb);
		if (floatRand(0, 1) < nInfectedAgents)
		{
			agents[i].state = 1;
		}
	}
}

void InitPlaces(Place* places)
{
	int i;
	int size = nPlacesCPU;
#pragma omp parallel for shared(size, places), private(i)
	for (i = 0; i < size; i++)
	{
		places[i].contactFactor = floatRand(0, maxExtavertizm);
	}
}

void SimulationCPU(int* healthy, int* infected, int* convalescent, int* dead)
{
	Agent* agents = new Agent[nAgents];
	Disease disease;
	Place* places = new Place[nPlacesCPU];
	std::srand(std::time(NULL));

	disease.duration = Dduration;
	disease.contagiousness = Dcontagiousness;
	InitAgents(agents);
	InitPlaces(places);
	
	
	for (int i = 0; i < simTime; i++)
	{
		auto t1 = std::chrono::steady_clock::now();
		printf("Day %d\\%d: \n", i + 1, simTime);
		for (int dayPart = 0; dayPart < nJourney; dayPart++)
		{
			InfectionTest(agents, disease, places);
		}
		MaskingVaccinAgents(agents);
		TestDeath(agents);
		DiseaseMutuation(disease);
		UpdateAgents(agents);

		SumOutputs(agents, healthy, infected, convalescent, dead, i);
		auto t2 = std::chrono::steady_clock::now();
		std::cout << "One Loop Time [ms]:" << (float)std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1000000 << "\n";
	}
	
	delete[] agents;
	delete[] places;
}