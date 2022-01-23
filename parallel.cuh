#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Place.cuh"
#include "Disease.cuh"
#include "Agent.cuh"
#include "parameters.h"
#include "randomf.h"

typedef unsigned int uint;
