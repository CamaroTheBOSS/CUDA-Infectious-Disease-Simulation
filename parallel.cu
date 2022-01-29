#include "parallel.cuh"
#include "CPU.h"
#include <chrono>
#include <string>

__device__ void SwapDevice(Agent* agent1, Agent* agent2)
{
	Agent temp = *agent1;
	*agent1 = *agent2;
	*agent2 = temp;
}

__device__ bool testAgent(Agent agent, float infProb, curandState state, SimulationParameters* sim) //test for coweed, if 1: positive, if 0: negative
{
	float x = curand_uniform(&state);


	float ressistanceComponent = agent.ressistance * infProb;
	float vacRessist = infProb * agent.vacRessist / sim[0].vaccinTimex;

	infProb -= ressistanceComponent + vacRessist;
	//printf("%f\n", infProb);
	if (x > infProb)
	{
		return 0;
	}
	return 1;
}

__device__ float calculateInfectionProbability(Agent agents[], Disease disease, Place place, int border, bool even, int blockidx, int blocksize, SimulationParameters* sim)
{
	float infectionprob;

	// Infection Variables (increase infection probability)
	float infectionComponent = 0; // aritmetic average of probabilities for infecting someone other by each infected Agent
	float nAgentsComponent = 0; // infected agents to all the agents in given place ratio
	float diseaseComponent = 0; // disease influence
	float placeComponent = 0; // more contactable places gives bigger chance for getting infected

	int nInfected = 0;
	if (even)
	{
		for (int j = 0; j < border; j++)
		{			
			if (agents[j].state == 1)
			{
				
				infectionComponent += agents[j].infectProb;
				nInfected++;
				if (agents[j].masked)
				{
					infectionComponent -= agents[j].infectProb * sim[0].maskEffectivnessx;
				}
			}			
		}
		//TODO BETTER MODEL
		infectionComponent /= (nInfected + 1);
		nAgentsComponent = infectionComponent * (float)nInfected / (float)(border + 1);
		diseaseComponent = disease.contagiousness * nInfected / (float)(border + 1);
		placeComponent = place.contactFactor * nInfected / (float)(border + 1);
		infectionprob = infectionComponent + nAgentsComponent + diseaseComponent + placeComponent;
		//printf("infComp: %f, nAgentsComp: %f, diseaseComp: %f, placeComp: %f\n", infectionComponent, nAgentsComponent, diseaseComponent, placeComponent);
	}
	else
	{
		for (int j = border; j < blocksize; j++)
		{
			if (agents[j].state == 1)
			{
				infectionComponent += agents[j].infectProb;
				nInfected++;
				if (agents[j].masked)
				{
					infectionComponent -= agents[j].infectProb * sim[0].maskEffectivnessx;
				}
			}
		}
		//TODO BETTER MODEL
		infectionComponent /= (nInfected + 1);
		nAgentsComponent = nInfected / (blocksize - border + 1);
		diseaseComponent = disease.contagiousness * nInfected / (blocksize - border + 1);
		placeComponent = place.contactFactor * nInfected / (blocksize - border + 1);
		infectionprob = infectionComponent + nAgentsComponent + diseaseComponent + placeComponent;
	}

	if (infectionprob > 1)
	{
		infectionprob = 1;
	}
	//printf("%f\n", infectionprob);
	return infectionprob;
}

__global__ void InfectionTest(Agent* agents, Disease* disease, Place* places, curandState* states, int* borders, int BlockSize, SimulationParameters* sim)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	extern __shared__ Agent sharedAgents[];
	extern __shared__ Disease sharedDisease;
	extern __shared__ Place sharedPlaces[2];
	__shared__ int sharedBorders;

	sharedAgents[threadIdx.x] = agents[i];
	sharedDisease = disease[0];
	sharedPlaces[0] = places[blockIdx.x * 2];
	sharedPlaces[1] = places[blockIdx.x * 2 + 1];
	sharedBorders = borders[blockIdx.x];
	__syncthreads();
	
	if (i < sim[0].nAgentsx)
	{
		float infProb;
		int nThreads;
		if (threadIdx.x < borders[blockIdx.x])
		{
			nThreads = BlockSize - borders[blockIdx.x];
			infProb = calculateInfectionProbability(sharedAgents, sharedDisease, sharedPlaces[0], sharedBorders, true, blockIdx.x, BlockSize, sim);
		}
		else
		{
			nThreads = borders[blockIdx.x];
			infProb = calculateInfectionProbability(sharedAgents, sharedDisease, sharedPlaces[1], sharedBorders, false, blockIdx.x, BlockSize, sim);
		}
		if (sharedAgents[threadIdx.x].state == 0)
		{
			if (testAgent(agents[i], infProb, states[i], sim))
			{
				agents[i].state = 1;
				agents[i].sickDaysLeft = sim[0].durationx;
				//printf("Im sick :(\n");
			}			
		}
	}
	
}

__global__ void BitonicShuffler(Agent* agents, curandState* states, SimulationParameters* sim) //Sorting random numbers
{
	int i, ixj;
	i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < sim[0].nAgentsx)
	{
		for (int k = 2; k <= sim[0].nAgentsx; k <<= 1)
		{
			for (int j = k >> 1; j > 0; j = j >> 1)
			{
				ixj = i ^ j;
				agents[i].randN = curand_uniform(&states[i]);
				__syncthreads();

				if ((ixj) > i)
				{
					if ((i & k) == 0)
					{
						if (agents[i].randN > agents[ixj].randN)
						{
							SwapDevice(&agents[i], &agents[ixj]);
						}
					}
					else if ((i & k) != 0)
					{
						if (agents[i].randN < agents[ixj].randN)
						{
							SwapDevice(&agents[i], &agents[ixj]);
						}
					}
				}
				__syncthreads();
			}
		}
	}
}

__global__ void DefineBorders(int* borders, int BlockSize, curandState* states)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	borders[i] = curand_uniform(&states[i]) * BlockSize;
}

__global__ void MaskingAgents(Agent* agents, curandState* states, SimulationParameters* sim)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < sim[0].nAgentsx)
	{
		float x = curand_uniform(&states[i]);
		if (x < agents[i].swapMaskProb)
		{
			if (agents[i].masked)
				agents[i].masked = false;
			else
				agents[i].masked = true;
		}
	}
}

__global__ void VaccinatingAgents(Agent* agents, curandState* states, SimulationParameters* sim)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < sim[0].nAgentsx)
	{
		float x = curand_uniform(&states[i]);
		if ((x < agents[i].vaccinWill) && (agents[i].vacRessist == 0))
		{
			agents[i].vacRessist = sim[0].vaccinTimex;
		}
	}
}

__global__ void testDeath(Agent* agents, curandState* states, SimulationParameters* sim)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if ((i < sim[0].nAgentsx) && (agents[i].state == 1))
	{
		float x = curand_uniform(&states[i]);
		if (x < agents[i].deathProb)
		{
			agents[i].state = 3; //death
			//printf("Im dead :(\n");
		}
	}
}

__global__ void UpdateAgents(Agent* agents, SimulationParameters* sim)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < sim[0].nAgentsx)
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

__global__ void diseaseMutuation(Disease* disease, curandState* states, SimulationParameters* sim)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	float x = curand_uniform(&states[i]);
	if (x < sim[0].mutuationProbx)
	{
		x = curand_uniform(&states[i]) * 2 - 1;
		disease[0].contagiousness += x * sim[0].mutuationIntensityx * disease[0].contagiousness;
	}
}

__global__ void InitSeeds(curandState* states, long int seed, SimulationParameters* sim)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < sim[0].nAgentsx)
	{
		seed += i;
		curand_init(seed, i, 0, &states[i]);
	}
}

__global__ void InitAgents(Agent* agents, curandState* states, SimulationParameters* sim)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	agents[i].deathProb = curand_uniform(&states[i]) * sim[0].maxDeathProbbx;
	agents[i].ressistance = curand_uniform(&states[i]) * sim[0].maxAgentRessistancex;
	agents[i].infectProb = curand_uniform(&states[i]) * sim[0].maxInfectProbx;
	agents[i].vaccinWill = curand_uniform(&states[i]) * sim[0].maxVaccinationProbx;
	agents[i].swapMaskProb = curand_uniform(&states[i]) * sim[0].maxMaskSwapProbx;
	float x = curand_uniform(&states[i]);
	if (x < sim[0].nInfectedAgentsProcentx)
	{
		agents[i].state = 1;
		agents[i].sickDaysLeft = sim[0].durationx;
		//printf("Start Root :(\n");
	}
}

__global__ void InitDiseasee(Disease* disease, SimulationParameters* sim)
{
	disease[0].contagiousness = sim[0].contagiousnesx;
	disease[0].duration = sim[0].durationx;
}

__global__ void InitPlaces(Place* places, curandState* states, SimulationParameters* sim)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	places[2 * i].contactFactor = curand_uniform(&states[i]) * sim[0].maxContactFactorx;
	places[2 * i + 1].contactFactor = curand_uniform(&states[i]) * sim[0].maxContactFactorx;
}

__global__ void SumHealthyAgents(Agent* agents, int* healthy, int BlockSize, int day, SimulationParameters* sim)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	extern __shared__ int sharedHealthyAgents[];
	sharedHealthyAgents[threadIdx.x] = 0;
	__syncthreads();

	if (i < sim[0].nAgentsx)
	{
		if (agents[i].state == 0)
		{
			sharedHealthyAgents[threadIdx.x] = 1;
		}
		if ((i % BlockSize) == 0)
		{
			for (int j = 1; j < BlockSize; j++)
			{
				sharedHealthyAgents[threadIdx.x] += sharedHealthyAgents[j];
			}
			atomicAdd(&healthy[day], sharedHealthyAgents[threadIdx.x]);
		}
		if (i == 0)
		{
			printf("Healthy: %d\n", healthy[day]);
		}
	}
}

__global__ void SumInfectedAgents(Agent* agents, int* infected, int BlockSize, int day, SimulationParameters* sim)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	extern __shared__ int sharedInfectedAgents[];
	sharedInfectedAgents[threadIdx.x] = 0;
	__syncthreads();

	if (i < sim[0].nAgentsx)
	{
		if (agents[i].state == 1)
		{
			sharedInfectedAgents[threadIdx.x] = 1;
		}
		if ((i % BlockSize) == 0)
		{
			for (int j = 1; j < BlockSize; j++)
			{
				sharedInfectedAgents[threadIdx.x] += sharedInfectedAgents[j];
			}
			atomicAdd(&infected[day], sharedInfectedAgents[threadIdx.x]);
		}
		if (i == 0)
		{
			printf("Infected: %d\n", infected[day]);
		}
	}
}

__global__ void SumConvalescentAgents(Agent* agents, int* convalescent, int BlockSize, int day, SimulationParameters* sim)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	extern __shared__ int sharedConvalescentAgents[];
	sharedConvalescentAgents[threadIdx.x] = 0;
	__syncthreads();

	if (i < sim[0].nAgentsx)
	{
		if (agents[i].state == 2)
		{
			sharedConvalescentAgents[threadIdx.x] = 1;
		}
		if ((i % BlockSize) == 0)
		{
			for (int j = 1; j < BlockSize; j++)
			{
				sharedConvalescentAgents[threadIdx.x] += sharedConvalescentAgents[j];
			}
			atomicAdd(&convalescent[day], sharedConvalescentAgents[threadIdx.x]);
		}
		if (i == 0)
		{
			printf("Convalescent: %d\n", convalescent[day]);
		}
	}
}

__global__ void SumDiedAgents(Agent* agents, int* died, int BlockSize, int day, SimulationParameters* sim)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	extern __shared__ int sharedDiedAgents[];
	sharedDiedAgents[threadIdx.x] = 0;
	__syncthreads();

	if (i < sim[0].nAgentsx)
	{
		if (agents[i].state == 3)
		{
			sharedDiedAgents[threadIdx.x] = 1;
		}
		if ((i % BlockSize) == 0)
		{
			for (int j = 1; j < BlockSize; j++)
			{
				sharedDiedAgents[threadIdx.x] += sharedDiedAgents[j];
			}
			atomicAdd(&died[day], sharedDiedAgents[threadIdx.x]);
		}
		if (i == 0)
		{
			printf("Dead: %d\n", died[day]);
		}
	}
}

__host__ void GetDeviceParameters(uint& BlockNum, uint& BlockSize, SimulationParameters* sim)
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	BlockSize = deviceProp.maxThreadsPerBlock;
	BlockNum = ceil(float(sim[0].nAgentsx) / float(BlockSize));
	if (BlockSize > sim[0].nAgentsx)
		BlockSize = sim[0].nAgentsx;
}

__host__ void PrintOutputs(int* healthy, int* infected, int* convalescent, int* died, SimulationParameters* sim)
{
	for (int i = 0; i < sim[0].simTimex; i++)
	{
		printf("H: %d, I: %d, C: %d, D: %d\n", healthy[i], infected[i], convalescent[i], died[i]);
	}
}

__host__ void SimulationGPU(int* healthy, int* infected, int* convalescent, int* died, SimulationParameters* sim)
{
	// Allocating simulation parameters at GPU
	SimulationParameters* simDev;
	size_t simSize = sizeof(SimulationParameters);
	cudaMalloc((void**)&simDev, simSize);
	cudaMemcpy(simDev, sim, simSize, cudaMemcpyHostToDevice);

	// Get device parameters to send data asynchronously and specify number of blocks and threads for each block
	int device = cudaGetDevice(&device);
	uint BlockNum = 0;
	uint BlockSize = 0;
	GetDeviceParameters(BlockNum, BlockSize, sim);
	printf("BlockNum, BlockSize: %d, %d\n", BlockNum, BlockSize);

	// Initialize randomness for each thread
	std::srand(std::time(NULL));
	curandState* states;
	cudaMalloc(&states, sizeof(curandState) * sim[0].nAgentsx);
	InitSeeds << <BlockNum, BlockSize >> > (states, std::clock(), simDev);


	//Malloc interior variables
	int* borders;
	cudaMalloc((void**)&borders, sizeof(int) * BlockNum);


	// Malloc memory for arrays with information about infected, healthy and convalescent agents number (Outputs)
	int* infectedDev;
	int* healthyDev;
	int* convalescentDev;
	int* deadDev;
	size_t OutputSize = sizeof(int) * sim[0].simTimex;
	cudaMalloc((void**)&infectedDev, OutputSize);
	cudaMalloc((void**)&healthyDev, OutputSize);
	cudaMalloc((void**)&convalescentDev, OutputSize);
	cudaMalloc((void**)&deadDev, OutputSize);
	
	
	// Allocate agents, places and disease in device memory
	Agent* agents;
	Disease* disease;
	Place* places;	
	size_t AgentSize = sizeof(Agent) * sim[0].nAgentsx;
	size_t DiseaseSize = sizeof(Disease);
	size_t PlacesSize = sizeof(Place) * 2 * BlockNum;	
	cudaMalloc((void**)&agents, AgentSize);
	cudaMalloc((void**)&disease, DiseaseSize);
	cudaMalloc((void**)&places, PlacesSize);

	InitAgents << <BlockNum, BlockSize>> > (agents, states, simDev);
	InitDiseasee << <1, 1 >> > (disease, simDev);
	InitPlaces << <BlockNum, 1 >> > (places, states, simDev);
	cudaDeviceSynchronize();

	SumHealthyAgents << <BlockNum, BlockSize, BlockSize * sizeof(int) >> > (agents, healthyDev, BlockSize, 0, simDev);
	cudaDeviceSynchronize();
	SumInfectedAgents << <BlockNum, BlockSize, BlockSize * sizeof(int) >> > (agents, infectedDev, BlockSize, 0, simDev);
	cudaDeviceSynchronize();
	SumConvalescentAgents << <BlockNum, BlockSize, BlockSize * sizeof(int) >> > (agents, convalescentDev, BlockSize, 0, simDev);
	cudaDeviceSynchronize();
	SumDiedAgents << <BlockNum, BlockSize, BlockSize * sizeof(int) >> > (agents, deadDev, BlockSize, 0, simDev);
	cudaDeviceSynchronize();

	
	size_t sharedSize = BlockSize * sizeof(Agent) + sizeof(Disease) + sizeof(Place) * 2 + sizeof(int);
	auto t3 = std::chrono::steady_clock::now();
	for (int i = 1; i < simTime + 1; i++)
	{
		auto t1 = std::chrono::steady_clock::now();
		printf("Day %d\\%d: \n", i, sim[0].simTimex);
		for (int dayPart = 0; dayPart < nJourney; dayPart++)
		{
			DefineBorders << <BlockNum, 1 >> > (borders, BlockSize, states);
			cudaDeviceSynchronize();
			InfectionTest << <BlockNum, BlockSize, sharedSize >> > (agents, disease, places, states, borders, BlockSize, simDev);
			cudaDeviceSynchronize();
			BitonicShuffler << <BlockNum, BlockSize >> > (agents, states, simDev);
			cudaDeviceSynchronize();
		}
		//change states
		MaskingAgents << <BlockNum, BlockSize >> > (agents, states, simDev);
		VaccinatingAgents << <BlockNum, BlockSize >> > (agents, states, simDev);
		cudaDeviceSynchronize();
		testDeath << <BlockNum, BlockSize >> > (agents, states, simDev);
		cudaDeviceSynchronize();
		diseaseMutuation << <1, 1 >> > (disease, states, simDev);
		cudaDeviceSynchronize();
		UpdateAgents << <BlockNum, BlockSize >> > (agents, simDev);
		cudaDeviceSynchronize();
		
		//Get outputs
		SumHealthyAgents << <BlockNum, BlockSize, BlockSize * sizeof(int) >> > (agents, healthyDev, BlockSize, i, simDev);
		cudaDeviceSynchronize();
		SumInfectedAgents << <BlockNum, BlockSize, BlockSize * sizeof(int) >> > (agents, infectedDev, BlockSize, i, simDev);
		cudaDeviceSynchronize();
		SumConvalescentAgents << <BlockNum, BlockSize, BlockSize * sizeof(int) >> > (agents, convalescentDev, BlockSize, i, simDev);
		cudaDeviceSynchronize();
		SumDiedAgents << <BlockNum, BlockSize, BlockSize * sizeof(int) >> > (agents, deadDev, BlockSize, i, simDev);
		cudaDeviceSynchronize();

		auto t2 = std::chrono::steady_clock::now();
		std::cout << "One Loop Time [ms]:" << (float)std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1000000 << "\n";
		printf("\n");
	}
	cudaDeviceSynchronize();
	auto t4 = std::chrono::steady_clock::now();

	cudaMemcpy(healthy, healthyDev, OutputSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(infected, infectedDev, OutputSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(convalescent, convalescentDev, OutputSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(died, deadDev, OutputSize, cudaMemcpyDeviceToHost);

	cudaFree(agents); cudaFree(disease); cudaFree(infectedDev); cudaFree(healthyDev); cudaFree(convalescentDev); cudaFree(deadDev); cudaFree(states); cudaFree(places);
}

__host__ void SetSimParameters(SimulationParameters &sim)
{
	// Simulation parameters ------------------------------------------
	sim.nAgentsx = pow(2, 11);
	sim.simTimex = 365;
	sim.vaccinTimex = 365;
	sim.nJourneyx = 3;
	sim.nInfectedAgentsProcentx = 0.01;
	sim.maskEffectivnessx = 0.5;

	// Agents boundaries ----------------------------------------------
	sim.maxDeathProbbx = 0.01;
	sim.maxInfectProbx = 0.05;
	sim.maxAgentRessistancex = 0.7;
	sim.maxMaskSwapProbx = 0.1;
	sim.maxVaccinationProbx = 0.2;

	// Disease parameters ----------------------------------------------
	sim.mutuationProbx = 0.005;
	sim.mutuationIntensityx = 0.1;
	sim.contagiousnesx = 0.05;
	sim.durationx = 14;

	// Places parameters -----------------------------------------------
	sim.maxContactFactorx = 0.1;
	sim.nPlacesCPUx = 2;
}

int main()
{
	bool GPU_ON = false; //if true, simulation will be turned on on the GPU
	SimulationParameters* SIM = new SimulationParameters;
	SetSimParameters(SIM[0]);

	int* infected = new int[SIM[0].simTimex + 1];
	int* healthy = new int[SIM[0].simTimex + 1];
	int* convalescent = new int[SIM[0].simTimex + 1];
	int* dead = new int[SIM[0].simTimex + 1];
	
	if (GPU_ON)
	{
		SimulationGPU(healthy, infected, convalescent, dead, SIM);
	}
	else
	{		
		SimulationCPU(healthy, infected, convalescent, dead, SIM);
	}

	PrintOutputs(healthy, infected, convalescent, dead, SIM);

	delete[] infected;
	delete[] healthy;
	delete[] convalescent;
	delete[] dead;
	delete SIM;
	return 0;
}