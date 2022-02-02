#include "randomf.h"

struct Place
{
	//unsigned short int cap = 0;      //max number of people which can be in place
	//unsigned short int size = 0;    //number of people who currently are in given place
	float contactFactor = 0;       //increase risk of beeing infected (0, 1) (bigger contactFactor means that place gives better enviroment for beeing infected e.g. a lot of people close to each other - festival or concert)
	//int* residents = nullptr;	  //Contains information about Agents in this place
};