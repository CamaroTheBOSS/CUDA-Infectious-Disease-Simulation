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

__device__ bool testAgent(Agent agent, float infProb, curandState state) //test for coweed, if 1: positive, if 0: negative
{
	float x = curand_uniform(&state);


	float ressistanceComponent = agent.ressistance * infProb;
	float vacRessist = infProb * agent.vacRessist / vaccinTime;

	infProb -= ressistanceComponent + vacRessist;
	//printf("%f\n", infProb);
	if (x > infProb)
	{
		return 0;
	}
	return 1;
}

__device__ float calculateInfectionProbability(Agent agents[], Disease disease, Place place, int border, bool even, int blockidx, int blocksize)
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
					infectionComponent -= agents[j].infectProb * maskEffectivness;
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
					infectionComponent -= agents[j].infectProb * 0.5;
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

__global__ void InfectionTest(Agent* agents, Disease* disease, Place* places, curandState* states, int* borders, int BlockSize)
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
	
	if (i < nAgents)
	{
		float infProb;
		int nThreads;
		if (threadIdx.x < borders[blockIdx.x])
		{
			nThreads = BlockSize - borders[blockIdx.x];
			infProb = calculateInfectionProbability(sharedAgents, sharedDisease, sharedPlaces[0], sharedBorders, true, blockIdx.x, BlockSize);
		}
		else
		{
			nThreads = borders[blockIdx.x];
			infProb = calculateInfectionProbability(sharedAgents, sharedDisease, sharedPlaces[1], sharedBorders, false, blockIdx.x, BlockSize);
		}
		if (sharedAgents[threadIdx.x].state == 0)
		{
			if (testAgent(agents[i], infProb, states[i]))
			{
				agents[i].state = 1;
				agents[i].sickDaysLeft = Dduration;
				//printf("Im sick :(\n");
			}			
		}
	}
	
}

__global__ void BitonicShuffler(Agent* agents, curandState* states) //Sorting random numbers
{
	int i, ixj;
	i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < nAgents)
	{
		for (int k = 2; k <= nAgents; k <<= 1)
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

__global__ void MaskingAgents(Agent* agents, curandState* states)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < nAgents)
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

__global__ void VaccinatingAgents(Agent* agents, curandState* states)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < nAgents)
	{
		float x = curand_uniform(&states[i]);
		if ((x < agents[i].vaccinWill) && (agents[i].vacRessist == 0))
		{
			agents[i].vacRessist == vaccinTime;
		}
	}
}

__global__ void testDeath(Agent* agents, curandState* states)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if ((i < nAgents) && (agents[i].state == 1))
	{
		float x = curand_uniform(&states[i]);
		if (x < agents[i].deathProb)
		{
			agents[i].state = 3; //death
			//printf("Im dead :(\n");
		}
	}
}

__global__ void healingAgents(Agent* agents)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < nAgents)
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

__global__ void diseaseMutuation(Disease* disease, curandState* states)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	float x = curand_uniform(&states[i]);
	if (x < mutuationProb)
	{
		x = curand_uniform(&states[i]) * 2 - 1;
		disease[0].contagiousness += x * mutuationIntensity * disease[0].contagiousness;
	}
}

__global__ void InitSeeds(curandState* states, long int seed)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < nAgents)
	{
		seed += i;
		curand_init(seed, i, 0, &states[i]);
	}
}

__global__ void InitAgents(Agent* agents, curandState* states)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	agents[i].deathProb = curand_uniform(&states[i]) * maxDeathProb;
	agents[i].ressistance = curand_uniform(&states[i]) * maxRessistanceParameter;
	agents[i].infectProb = curand_uniform(&states[i]) * maxInfectProb;
	agents[i].vaccinWill = curand_uniform(&states[i]) * maxVaccinProb;
	agents[i].swapMaskProb = curand_uniform(&states[i]) * maxSwapMaskProb;
	float x = curand_uniform(&states[i]);
	if (x < nInfectedAgents)
	{
		agents[i].state = 1;
		agents[i].sickDaysLeft = Dduration;
		//printf("Start Root :(\n");
	}
}

__global__ void InitDiseasee(Disease* disease)
{
	disease[0].contagiousness = Dcontagiousness;
	disease[0].duration = Dduration;
}

__global__ void InitPlaces(Place* places, curandState* states)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	places[2 * i].contactFactor = curand_uniform(&states[i]) * maxExtavertizm;
	places[2 * i + 1].contactFactor = curand_uniform(&states[i]) * maxExtavertizm;
}

__global__ void SumHealthyAgents(Agent* agents, int* healthy, int BlockSize, int day)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	extern __shared__ int sharedHealthyAgents[];
	sharedHealthyAgents[threadIdx.x] = 0;
	__syncthreads();

	if (i < nAgents)
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

__global__ void SumInfectedAgents(Agent* agents, int* infected, int BlockSize, int day)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	extern __shared__ int sharedInfectedAgents[];
	sharedInfectedAgents[threadIdx.x] = 0;
	__syncthreads();

	if (i < nAgents)
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

__global__ void SumConvalescentAgents(Agent* agents, int* convalescent, int BlockSize, int day)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	extern __shared__ int sharedConvalescentAgents[];
	sharedConvalescentAgents[threadIdx.x] = 0;
	__syncthreads();

	if (i < nAgents)
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

__global__ void SumDiedAgents(Agent* agents, int* died, int BlockSize, int day)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	extern __shared__ int sharedDiedAgents[];
	sharedDiedAgents[threadIdx.x] = 0;
	__syncthreads();

	if (i < nAgents)
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

__host__ void GetDeviceParameters(uint& BlockNum, uint& BlockSize)
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	BlockSize = deviceProp.maxThreadsPerBlock;
	BlockNum = ceil(float(nAgents) / float(BlockSize));
	if (BlockSize > nAgents)
		BlockSize = nAgents;
}

__host__ void PrintOutputs(int* healthy, int* infected, int* convalescent, int* died)
{
	for (int i = 0; i < simTime; i++)
	{
		printf("H: %d, I: %d, C: %d, D: %d\n", healthy[i], infected[i], convalescent[i], died[i]);
	}
}

__host__ void SimulationGPU(int* healthy, int* infected, int* convalescent, int* died)
{
	// Get device parameters to send data asynchronously and specify number of blocks and threads for each block
	int device = cudaGetDevice(&device);
	uint BlockNum = 0;
	uint BlockSize = 0;
	GetDeviceParameters(BlockNum, BlockSize);
	printf("BlockNum, BlockSize: %d, %d\n", BlockNum, BlockSize);

	// Initialize randomness for each thread
	std::srand(std::time(NULL));
	curandState* states;
	cudaMalloc(&states, sizeof(curandState) * nAgents);
	InitSeeds << <BlockNum, BlockSize >> > (states, std::clock());


	//Malloc interior variables
	int* borders;
	cudaMalloc((void**)&borders, sizeof(int) * BlockNum);


	// Malloc memory for arrays with information about infected, healthy and convalescent agents number (Outputs)
	int* infectedDev;
	int* healthyDev;
	int* convalescentDev;
	int* deadDev;
	size_t OutputSize = sizeof(int) * simTime;
	cudaMalloc((void**)&infectedDev, OutputSize);
	cudaMalloc((void**)&healthyDev, OutputSize);
	cudaMalloc((void**)&convalescentDev, OutputSize);
	cudaMalloc((void**)&deadDev, OutputSize);
	
	
	// Allocate agents, places and disease in unified memory
	Agent* agents;
	Disease* disease;
	Place* places;
	size_t AgentSize = sizeof(Agent) * nAgents;
	size_t DiseaseSize = sizeof(Disease);
	size_t PlacesSize = sizeof(Place) * 2 * BlockNum;
	cudaMalloc((void**)&agents, AgentSize);
	cudaMalloc((void**)&disease, DiseaseSize);
	cudaMalloc((void**)&places, PlacesSize);
	

	InitAgents << <BlockNum, BlockSize>> > (agents, states);
	InitDiseasee << <1, 1 >> > (disease);
	InitPlaces << <BlockNum, 1 >> > (places, states);
	cudaDeviceSynchronize();

	SumHealthyAgents << <BlockNum, BlockSize, BlockSize * sizeof(int) >> > (agents, healthyDev, BlockSize, 0);
	cudaDeviceSynchronize();
	SumInfectedAgents << <BlockNum, BlockSize, BlockSize * sizeof(int) >> > (agents, infectedDev, BlockSize, 0);
	cudaDeviceSynchronize();
	SumConvalescentAgents << <BlockNum, BlockSize, BlockSize * sizeof(int) >> > (agents, convalescentDev, BlockSize, 0);
	cudaDeviceSynchronize();
	SumDiedAgents << <BlockNum, BlockSize, BlockSize * sizeof(int) >> > (agents, deadDev, BlockSize, 0);
	cudaDeviceSynchronize();

	
	size_t sharedSize = BlockSize * sizeof(Agent) + sizeof(Disease) + sizeof(Place) * 2 + sizeof(int);
	auto t3 = std::chrono::steady_clock::now();
	for (int i = 1; i < simTime + 1; i++)
	{
		auto t1 = std::chrono::steady_clock::now();
		printf("Day %d\\%d: \n", i, simTime);
		for (int dayPart = 0; dayPart < nJourney; dayPart++)
		{
			DefineBorders << <BlockNum, 1 >> > (borders, BlockSize, states);
			cudaDeviceSynchronize();
			InfectionTest << <BlockNum, BlockSize, sharedSize >> > (agents, disease, places, states, borders, BlockSize);
			cudaDeviceSynchronize();

			BitonicShuffler << <BlockNum, BlockSize >> > (agents, states);
			cudaDeviceSynchronize();
			std::cout << cudaGetErrorString << "\n";
		}
		//change states
		MaskingAgents << <BlockNum, BlockSize >> > (agents, states);
		VaccinatingAgents << <BlockNum, BlockSize >> > (agents, states);
		cudaDeviceSynchronize();
		testDeath << <BlockNum, BlockSize >> > (agents, states);
		cudaDeviceSynchronize();
		diseaseMutuation << <1, 1 >> > (disease, states);
		cudaDeviceSynchronize();
		healingAgents << <BlockNum, BlockSize >> > (agents);
		cudaDeviceSynchronize();
		
		//Get outputs
		SumHealthyAgents << <BlockNum, BlockSize, BlockSize * sizeof(int) >> > (agents, healthyDev, BlockSize, i);
		cudaDeviceSynchronize();
		SumInfectedAgents << <BlockNum, BlockSize, BlockSize * sizeof(int) >> > (agents, infectedDev, BlockSize, i);
		cudaDeviceSynchronize();
		SumConvalescentAgents << <BlockNum, BlockSize, BlockSize * sizeof(int) >> > (agents, convalescentDev, BlockSize, i);
		cudaDeviceSynchronize();
		SumDiedAgents << <BlockNum, BlockSize, BlockSize * sizeof(int) >> > (agents, deadDev, BlockSize, i);
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

int main()
{
	bool GPU_ON = true; //if true, simulation will be turned on on the GPU

	int* infected = new int[simTime+1];
	int* healthy = new int[simTime+1];
	int* convalescent = new int[simTime+1];
	int* dead = new int[simTime+1];
	int startInfections = 0;
	
	if (GPU_ON)
	{
		SimulationGPU(healthy, infected, convalescent, dead);
	}
	else
	{		
		SimulationCPU(healthy, infected, convalescent, dead);
	}

	PrintOutputs(healthy, infected, convalescent, dead);

	delete[] infected;
	delete[] healthy;
	delete[] convalescent;
	delete[] dead;
	return 0;
}