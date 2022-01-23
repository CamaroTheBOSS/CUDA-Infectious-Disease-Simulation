#include <stdlib.h>
#include <time.h>

int intRand(int fMin, int fMax)
{
	return rand() % (fMax - fMin) + fMin;
}

float floatRand(float fMin, float fMax)
{
	float f = (float)rand() / RAND_MAX;
	return fMin + f * (fMax - fMin);
}
