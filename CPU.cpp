#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <chrono>
#include <set>
#include <algorithm>
#include <iterator>

#include "Place.cuh"
#include "Disease.cuh"
#include "Agent.cuh"
#include "randomf.h"
#include "SimulationParameters.cuh"

void SumOutputs(Agent* agents, int* healthy, int* infected, int* convalescent, int* dead, int day, SimulationParameters* sim)
{
	int i;
	int size = sim[0].nAgentsx;
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

void updateAgents(Agent* agents, SimulationParameters* sim)
{
	int i;
	int size = sim[0].nAgentsx;
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

void DiseaseMutuation(Disease& disease, SimulationParameters* sim)
{
	float x = floatRand(0, 1);
	if (x < sim[0].mutuationProbx)
	{
		x = floatRand(-sim[0].mutuationIntensityx, sim[0].mutuationIntensityx);
		disease.contagiousness += x * disease.contagiousness;
	}
}

void TestDeath(Agent* agents, SimulationParameters* sim)
{
	int i;
	int size = sim[0].nAgentsx;
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

void MaskingVaccinAgents(Agent* agents, SimulationParameters* sim)
{
	int size = sim[0].nAgentsx;
	int i;
	float x;
#pragma omp parallel for shared(size, agents, sim), private(i, x)
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
			agents[i].vacRessist == sim[0].vaccinTimex;
		}
	}
}

float TestAgent(Agent agent, float infProb, SimulationParameters* sim)
{
	float x = floatRand(0, 1);

	float ressistanceComponent = agent.ressistance * infProb;
	float vacRessist = infProb * agent.vacRessist / sim[0].vaccinTimex;
	infProb -= ressistanceComponent + vacRessist;
	if (x > infProb)
	{
		return 0;
	}
	return 1;
}

void InfectionTest(Agent* agents, Disease disease, Place* places, SimulationParameters* sim)
{
	int i, j;
	int size = sim[0].nAgentsx;
	int placeCap = sim[0].nAgentsx / sim[0].nPlacesCPUx;

	int firstIdx, lastIdx, placeIdx, nInfected;
	float infectionComponent, nAgentsComponent, diseaseComponent, placeComponent, infectionprob;


#pragma omp parallel for shared(size, placeCap, agents, places, disease, sim), private(i, j, firstIdx, lastIdx, placeIdx, nInfected, infectionComponent, nAgentsComponent, diseaseComponent, infectionprob)
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
					infectionComponent -= agents[j].infectProb * sim[0].maskEffectivnessx;
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
			if (TestAgent(agents[i], infectionprob, sim))
			{
				agents[i].state = 1;
				agents[i].sickDaysLeft = sim[0].durationx;
				//printf("Im sick :(\n");
			}
		}
	}
}

int InitAgents(Agent* agents, SimulationParameters* sim)
{
	int i;
	int size = sim[0].nAgentsx;
	float x;
#pragma omp parallel for shared(size, agents, sim), private(i, x)
	for (i = 0; i < size; i++)
	{
		agents[i].deathProb = floatRand(0, sim[0].maxDeathProbbx);
		agents[i].infectProb = floatRand(0, sim[0].maxInfectProbx);
		agents[i].ressistance = floatRand(0, sim[0].maxAgentRessistancex);
		agents[i].swapMaskProb = floatRand(0, sim[0].maxMaskSwapProbx);
		agents[i].vaccinWill = floatRand(0, sim[0].maxVaccinationProbx);
		x = floatRand(0, 1);
		if (x < sim[0].nInfectedAgentsProcentx)
		{
			agents[i].state = 1;
			agents[i].sickDaysLeft = sim[0].durationx;
		}
	}
}

void InitPlaces(Place* places, SimulationParameters* sim)
{
	int i;
	int size = sim[0].nPlacesCPUx;
#pragma omp parallel for shared(size, places, sim), private(i)
	for (i = 0; i < size; i++)
	{
		std::srand(omp_get_thread_num() * 1000 + std::time(NULL));
		places[i].contactFactor = floatRand(0, sim[0].maxContactFactorx);
	}
}

void InitSeeds(unsigned int* seeds)
{
	int my_thread_id;
	unsigned int seed;
#pragma omp parallel private (seed, my_thread_id)
	{
		my_thread_id = omp_get_thread_num();

		//create seed on thread using current time
		unsigned int seed = (unsigned)time(NULL);

		//munge the seed using our thread number so that each thread has its
		//own unique seed, therefore ensuring it will generate a different set of numbers
		seeds[my_thread_id] = (seed & 0xFFFFFFF0) | (my_thread_id + 1);
	}
}

void Swap(Agent* agent1, Agent* agent2)
{
	Agent temp = *agent1;
	*agent1 = *agent2;
	*agent2 = temp;
}

int DivideArray(Agent* agents, int minIdx, int maxIdx)
{
	Agent pivot = agents[maxIdx];
	int i = minIdx - 1;

	for (int j = minIdx; j < maxIdx; j++)
	{
		if (agents[j].randN < pivot.randN)
		{
			i++;
			Swap(&agents[i], &agents[j]);
		}
	}
	Swap(&agents[i + 1], &agents[maxIdx]);
	return (i + 1);
}

void QuickSortShuffler(Agent* agents, int minIdx, int maxIdx)
{
	int i;
#pragma omp parallel for shared(agents, maxIdx), private(i)
	for (i = 0; i < maxIdx; i++)
	{
		agents[i].randN = floatRand(0, 1);
	}

	int next;
	if (minIdx < maxIdx)
	{
		next = DivideArray(agents, minIdx, maxIdx);
		QuickSortShuffler(agents, minIdx, next - 1);
		QuickSortShuffler(agents, next + 1, maxIdx);
	}
}

template<typename S>
auto SelectRandomFromSet(const S& s, size_t n) 
{
	auto it = std::begin(s);
	// 'advance' the iterator n times
	std::advance(it, n);
	return it;
}

std::set<int> SetCpy(std::set<int> s)
{
	std::set<int> scpy;
	std::copy(s.begin(), s.end(), std::inserter(scpy, scpy.begin()));

	return scpy;
}

std::set<int> DefineSetOfInts(int maxInt)
{
	std::set<int> s;
	for (int i = 0; i < maxInt; i++)
	{
		s.insert(i);
	}

	return s;
}

void ShuffleWithSet(Agent* agents, std::set<int> s, SimulationParameters* sim)
{
	std::set<int> sCpy = SetCpy(s);

	for (int i = 0; i < sim[0].nAgentsx; i++)
	{
		auto r = rand() % s.size();
		auto n = *SelectRandomFromSet(s, r);

		Swap(&agents[i], &agents[n]);

		sCpy.erase(n);
	}
}

void PrintSlide(int k, int step)
{
	int i;
#pragma omp parallel for shared(k), private(i)
	for (i = 0; i <= k; i++)
	{
		printf("%c", 219);
	}
#pragma omp parallel for shared(step, k) private(i)
	for (i = 0; i < step - k; i++)
	{
		printf(" ");
	}
}

void PrintSliders(int* healthy, int* infected, int* convalescent, int* dead, int day, SimulationParameters* sim)
{
	int step = 30;
	int h = ceil(healthy[day] * step / sim[0].nAgentsx);
	int i = ceil(infected[day] * step / sim[0].nAgentsx);
	int c = ceil(convalescent[day] * step / sim[0].nAgentsx);
	int d = ceil(dead[day] * step / sim[0].nAgentsx);

	printf(" Healthy:       |                  ");
	gotoxy(18, 3);
	PrintSlide(h, step);
	printf("|\n");

	printf(" Infected:      |                  ");
	gotoxy(18, 4);
	PrintSlide(i, step);
	printf("|\n");

	printf(" Convalescent:  |                  ");
	gotoxy(18, 5);
	PrintSlide(c, step);
	printf("|\n");

	printf(" Dead:          |                  ");
	gotoxy(18, 6);
	PrintSlide(d, step);
	printf("|\n");

	gotoxy(1, 1);
}

void SimulationCPU(int* healthy, int* infected, int* convalescent, int* dead, SimulationParameters* sim)
{
	Agent* agents = new Agent[sim[0].nAgentsx];
	Disease disease;
	Place* places = new Place[sim[0].nPlacesCPUx];
	int maxThreads = omp_get_max_threads();
	unsigned int* seeds = new unsigned int[maxThreads];


	disease.duration = sim[0].durationx;
	disease.contagiousness = sim[0].contagiousnesx;
	std::set<int> set = DefineSetOfInts(sim[0].nAgentsx);
	InitSeeds(seeds);
	InitAgents(agents, sim);
	InitPlaces(places, sim);
	SumOutputs(agents, healthy, infected, convalescent, dead, 0, sim);


	for (int i = 1; i <= sim[0].simTimex; i++)
	{
		auto t1 = std::chrono::steady_clock::now();
		printf(" Day %d\\%d: \n", i, sim[0].simTimex);
		for (int dayPart = 0; dayPart < nJourney; dayPart++)
		{
			InfectionTest(agents, disease, places, sim);
			//QuickSortShuffler(agents, 0, sim[0].nAgentsx);
			ShuffleWithSet(agents, set, sim);
		}
		MaskingVaccinAgents(agents, sim);
		TestDeath(agents, sim);
		DiseaseMutuation(disease, sim);
		updateAgents(agents, sim);

		SumOutputs(agents, healthy, infected, convalescent, dead, i, sim);
		auto t2 = std::chrono::steady_clock::now();
		std::cout << " One Loop Time [ms]:" << (float)std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1000000 << "\n";
		PrintSliders(healthy, infected, convalescent, dead, i, sim);
	}

	delete[] agents;
	delete[] places;
	delete[] seeds;
}