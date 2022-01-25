#ifndef __CUDACC__ 
#define __CUDACC__
#endif

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <curand.h>

#include "Place.cuh"
#include "Disease.cuh"
#include "Agent.cuh"
#include "parameters.h"
#include "randomf.h"

typedef unsigned int uint;
