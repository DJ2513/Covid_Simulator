﻿#include <iostream>
#include <cmath>
#include <cstdlib>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

const int NUM_AGENTS = 1024;
const int MAX_DAYS = 30;
const int MAX_MOVEMENTS_PER_DAY = 10;
const float SIM_AREA_WIDTH = 500.0f;    
const float SIM_AREA_HEIGHT = 500.0f;   
const float MAX_LOCAL_MOVEMENT = 5.0f;
const float CONTAGION_DISTANCE = 1.0f;  

struct Agent {
    float contagionProb;         
    float externalContagionProb; 
    float mortalityProb;         
    float mobilityProb;          
    float shortMoveProb;         
    int incubationTime;          
    int recoveryTime;            
    int infectionStatus;
    float posX;
    float posY;
    int daysInfected;            
    int daysInQuarantine;        
};

float randFloat(float min, float max) {
    uniform_real_distribution<float> dist(min, max);
    return dist(gen);
}

__host__ void initializeAgents(Agent* agents) {
    random_device rd;
    mt19937 gen(rd());
    for (int i = 0; i < NUM_AGENTS; ++i) {
        agents[i].contagionProb = randFloat(0.02f, 0.03f, gen);
        agents[i].externalContagionProb = randFloat(0.02f, 0.03f, gen);
        agents[i].mortalityProb = randFloat(0.007f, 0.07f, gen);
        agents[i].mobilityProb = randFloat(0.3f, 0.5f, gen);
        agents[i].shortMoveProb = randFloat(0.7f, 0.9f, gen);
        agents[i].incubationTime = (randFloat(5.0f, 6.0f, gen) < 5.5f) ? 5 : 6;
        agents[i].recoveryTime = 14;
        agents[i].infectionStatus = 0; 
        agents[i].posX = randFloat(0.0f, SIM_AREA_WIDTH, gen);
        agents[i].posY = randFloat(0.0f, SIM_AREA_HEIGHT, gen);
        agents[i].daysInfected = 0;
        agents[i].daysInQuarantine = 0;
    }
}

// Rule 1: Contagion
// If an uninfected agent is near an infected neighbor, it may become infected.


// Rule 2: Mobility
// Agent moves either locally or long range based on probabilities.


// Rule 3: External Contagion
// An uninfected agent may become infected from outside the simulation.


// Rule 4: Incubation, Quarantine, and Recovery
// Update incubation days and transition to quarantine or recovery as needed.


// Rule 5: Fatal Cases
// In quarantine, an agent may die based on mortality probability.


// Sequential Simulation (CPU)
// This __host__ method calls all rules according to the simulation procedure.

// --------------------------
// Main Function
int main() {
    Agent* agents = new Agent[NUM_AGENTS];
    initializeAgents(agents);
    
    return 0;
}
