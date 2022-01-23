#include "parallel.cuh"



struct xd
{
	int xd1 = 1;
	int xd2 = 2;
	float xd3 = 3;

};

__global__ void NextDay()
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < nAgents)
	{
			
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

__host__ void InitAgents(Agent* &agents)
{
	for (int i = 0; i < nAgents; i++)
	{
		agents[i].deathProb = floatRand(0, maxDeathProb);
		agents[i].extrovertizm = floatRand(0, maxExtravertizmParameter);
		agents[i].getInfectedProb = floatRand(0, maxGetInfectedValue);
		agents[i].infectProb = floatRand(0, maxInfectProb);
	}
}

__host__ void InitPlaces(Place* &places)
{
	for (int i = 0; i < nPlaces; i++)
	{
		places[i].cap = avrCapacity + intRand(-standardDeviation, standardDeviation);
		places[i].contactFactor = floatRand(0, 1);
	}
}

__host__ void InitDisease(Disease* &disease)
{
		disease[0].contagiousness = Dcontagiousness;
		disease[0].duration = Dduration;
}

int main()
{
	srand(time(NULL));

	// Get device parameters to send data asynchronously and specify number of blocks and threads for each block
	int device = cudaGetDevice(&device);
	uint BlockNum = 0;
	uint BlockSize = 0;
	GetDeviceParameters(BlockNum, BlockSize);
	printf("BlockNum, BlockSize: %d, %d\n", BlockNum, BlockSize);


	// Malloc memory for arrays with information about infected, healthy and convalescent agents number (Outputs)
	uint* infected;
	uint* healthy;
	uint* convalescent;
	size_t OutputSize = sizeof(uint) * simTime;
	cudaMallocManaged(&infected, OutputSize);
	cudaMallocManaged(&healthy, OutputSize);
	cudaMallocManaged(&convalescent, OutputSize);
	// Make Prefetchs for outputs
	cudaMemPrefetchAsync(infected, OutputSize, device, NULL); // ptr, size_t, device, stream
	cudaMemPrefetchAsync(healthy, OutputSize, device, NULL);
	cudaMemPrefetchAsync(convalescent, OutputSize, device, NULL);
	

	// Allocate agents, places and disease in unified memory
	Agent* agents;
	Disease* disease;
	Place* places;
	size_t AgentSize = sizeof(Agent) * nAgents;
	size_t DiseaseSize = sizeof(Disease);
	size_t PlacesSize = sizeof(Place) * nPlaces;
	cudaMallocManaged(&agents, AgentSize);
	cudaMallocManaged(&disease, DiseaseSize);
	cudaMallocManaged(&places, PlacesSize);
	

	// Memory hints 
	cudaMemAdvise(agents, AgentSize, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId); // Start on CPU
	cudaMemAdvise(disease, DiseaseSize, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
	cudaMemAdvise(places, PlacesSize, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
	InitAgents(agents);
	InitPlaces(places);
	InitDisease(disease);
	// Prefetch agents to gpu 
	cudaMemPrefetchAsync(agents, AgentSize, device, NULL);
	cudaMemPrefetchAsync(disease, DiseaseSize, device, NULL);
	cudaMemPrefetchAsync(places, PlacesSize, device, NULL);

	for (int i = 0; i < simTime; i++)
	{
		NextDay << <BlockNum, BlockSize >> > ();
	}
	cudaDeviceSynchronize();

	//Get back the outputs
	cudaMemPrefetchAsync(infected, OutputSize, cudaCpuDeviceId);
	cudaMemPrefetchAsync(healthy, OutputSize, cudaCpuDeviceId);
	cudaMemPrefetchAsync(convalescent, OutputSize, cudaCpuDeviceId);


	return 0;
}