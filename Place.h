#include "parameters.h"
#include "randomf.h"

class Place
{
private:

	int cap = 0; //max number of people which can be in place
	int size = 0; //number of people who currently are in given place
	float contactFactor = 0; //increase risk of beeing infected (0, 1) (bigger contactFactor means that place gives better enviroment for beeing infected e.g. a lot of people close to each other - festival or concert)

public:

	Place::Place(void)
	{
		cap = avrCapacity + intRand(-standardDeviation, standardDeviation);
		contactFactor = floatRand(0, 1);
	}

	void placeReset()
	{
		size = 0;
	}

	float getParameter(int i)
	{
		if (i == 0)
		{
			return cap;
		}
		else
		{
			return contactFactor;
		}
	}

};

