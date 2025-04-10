#include <iostream>
#include <cmath>
#include <cstdlib>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "device_launch_parameters.h"

using namespace std;

const int NUM_AGENTS = 1024;
const int MAX_DAYS = 30;
const int MAX_MOVEMENTS_PER_DAY = 10;
const float SIM_AREA_WIDTH = 500.0f;    
const float SIM_AREA_HEIGHT = 500.0f;   
const float MAX_SHORT_MOVEMENT = 5.0f;
const float MAX_LONG_MOVEMENT = 20.0f;
const float CONTAGION_DISTANCE = 1.0f;  

struct Agent {
    float contagionProb;         
    float externalContagionProb; 
    float mortalityProb;         
    float mobilityProb;          
    float shortMoveProb;     
    float longMoveProb;
    int incubationTime;          
    int recoveryTime;            
    int infectionStatus;
    float posX;
    float posY;
    int daysInfected;            
    int daysInQuarantine;        
    int maxMovmentPerDay;
};

float randFloat(float min, float max, mt19937& gen) {
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
        agents[i].longMoveProb = randFloat(0.7f, 0.9f, gen);
        agents[i].incubationTime = (randFloat(5.0f, 6.0f, gen) < 5.5f) ? 5 : 6;
        agents[i].recoveryTime = 14;
        agents[i].infectionStatus = 0; 
        agents[i].posX = randFloat(0.0f, SIM_AREA_WIDTH, gen);
        agents[i].posY = randFloat(0.0f, SIM_AREA_HEIGHT, gen);
        agents[i].daysInfected = 0;
        agents[i].daysInQuarantine = 0;
        agents[i].maxMovmentPerDay = 10;

    }
}

__device__ curandState globalStates[NUM_AGENTS];

__global__ void setupStates() {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(clock(), id, 0, &globalStates[id]);
}

__device__ float gpuRandom() {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    return curand_uniform(&globalStates[id]);
}

// Rule 1: Contagion
// If an uninfected agent is near an infected neighbor, it may become infected.
__host__ void CPU_Rule1(Agent* agents, int num_agents, float contagiosDist) {
    for (int i = 0; i < num_agents; ++i) {
        for (int j = 0; j < num_agents; ++j) {
            if (i != j && agents[i].infectionStatus == 0 && agents[j].infectionStatus == 1) {
                // Calculate distance
                float dist = sqrt(pow(agents[i].posX - agents[j].posX, 2) + pow(agents[i].posY - agents[j].posY, 2));

                if (dist <= contagiosDist) {
                    float rand_val = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

                    if (rand_val <= agents[i].contagionProb) {
                        agents[i].infectionStatus = 1;
                    }
                }
            }
        }
    }
}


__device__ void GPU_Rule1(Agent* agents, int num_agents, int contagiosDist) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_agents) {
        for (int j = 0; j < num_agents; ++j) {
            if (i != j && agents[i].infectionStatus == 0 && agents[j].infectionStatus == 1) {
                // Calculate distance
                float dist = sqrt(pow(agents[i].posX - agents[j].posX, 2) + pow(agents[i].posY - agents[j].posY, 2));

                if (dist <= contagiosDist) {
                    float rand_val = 0.01;

                    if (rand_val <= agents[i].contagionProb) {
                        agents[i].infectionStatus = 1;
                    }
                }
            }
        }
    }
}


// Rule 2: Mobility
// Agent moves either locally or long range based on probabilities.
__host__ void CPU_Rule2(Agent* agents, int num_agents) {
    random_device rd;
    mt19937 gen(rd());

    for (int i = 0; i < num_agents; ++i) {
        if (agents[i].maxMovmentPerDay == 0) return;
        // Determine if the agent moves
        if (randFloat(0.0f, 1.0f, gen) <= agents[i].mobilityProb) {
            // Determine short or long move
            if (randFloat(0.0f, 1.0f, gen) <= agents[i].shortMoveProb) {
                // Short move
                float angle = randFloat(0.0f, 2.0f * 3.14, gen); // Random direction
                float dist = randFloat(0.0f, MAX_SHORT_MOVEMENT, gen);
                agents[i].posX += dist * cos(angle);
                agents[i].posY += dist * sin(angle);
            }
            else if (randFloat(0.0f, 1.0f, gen) <= agents[i].longMoveProb){
                // Long move
                float angle = randFloat(0.0f, 2.0f * 3.14, gen); // Random direction
                float dist = randFloat(0.0f, MAX_LONG_MOVEMENT, gen);
                agents[i].posX += dist * cos(angle);
                agents[i].posY += dist * sin(angle);
            }
            agents[i].posX = fmax(0.0f, fmin(agents[i].posX, SIM_AREA_WIDTH));
            agents[i].posY = fmax(0.0f, fmin(agents[i].posY, SIM_AREA_HEIGHT));
            agents[i].maxMovmentPerDay--;
        }
    }
}

__device__ void GPU_Rule2(Agent agent, int num_agents, float short_mov, float long_mov, float width, float height) {
    float rand_val = 0.5f;
    if (agent.maxMovmentPerDay == 0) return;
    if (rand_val <= agent.mobilityProb) {
        if (rand_val <= agent.shortMoveProb) {
            // Short move
            float angle = rand_val;
            float dist = short_mov * rand_val;
            agent.posX += dist * cosf(angle);
            agent.posY += dist * sinf(angle);
        }
        else {
            // Long move
            float angle = rand_val;
            float dist = long_mov * rand_val;
            agent.posX += dist * cosf(angle);
            agent.posY += dist * sinf(angle);
        }

        agent.posX = fmaxf(0.0f, fminf(agent.posX, width));
        agent.posY = fmaxf(0.0f, fminf(agent.posY, height));
        agent.maxMovmentPerDay--;
    }

}

// Rule 3: External Contagion
// An uninfected agent may become infected from outside the simulation.
__device__ void GPU_Rule3(Agent &agent) {
    if (agent.infectionStatus == 0) {
        unsigned int seed = threadIdx.x + blockIdx.x + clock();
        float prob = 0.01f;
        if (prob < agent.externalContagionProb) {
            agent.infectionStatus = 1;
            agent.daysInfected = 0;
        }
    }
}

__host__ void CPU_Rule3(Agent& agent, mt19937& gen) {
    if (agent.infectionStatus == 0) {
        if (randFloat(0.0f, 1.0f, gen) < agent.externalContagionProb) {
            agent.infectionStatus = 1;
            agent.daysInfected = 0;
        }
    }
}


// Rule 4: Incubation, Quarantine, and Recovery
// Update incubation days and transition to quarantine or recovery as needed.
__device__ void GPU_Rule4(Agent &agent) {
    if (agent.infectionStatus == 1) {
        agent.daysInfected++;
        if (agent.daysInfected >= agent.incubationTime) {
            agent.infectionStatus = -1;
            agent.daysInQuarantine = 0;
        }
    }
    else if (agent.infectionStatus == -1) {
        agent.daysInQuarantine++;
        if (agent.daysInQuarantine >= agent.recoveryTime) {
            agent.infectionStatus = 2;
        }
    }
}

__host__ void CPU_Rule4(Agent& agent) {
    if (agent.infectionStatus == 1) {
        agent.daysInfected++;
        if (agent.daysInfected >= agent.incubationTime) {
            agent.infectionStatus = -1;
            agent.daysInQuarantine = 0;
        }
    }
    else if (agent.infectionStatus == -1) {
        agent.daysInQuarantine++;
        if (agent.daysInQuarantine >= agent.recoveryTime) {
            agent.infectionStatus = 2;
        }
    }
}


// Rule 5: Fatal Cases
// In quarantine an agent may die based on mortality probability.
__device__ void GPU_Rule5(Agent &agent) {
    if (agent.infectionStatus == -1) {
        unsigned int seed = threadIdx.x + blockIdx.x + clock();
        float prob = 0.006;
        if (prob < agent.mortalityProb) {
            agent.infectionStatus = -2;
        }
    }
}

__host__ void CPU_Rule5(Agent& agent, mt19937& gen) {
    if (agent.infectionStatus == -1) {
        if (randFloat(0.0f, 1.0f, gen) < agent.mortalityProb) {
            agent.infectionStatus = -2;
        }
    }
}

// Sequential Simulation (CPU)
// This method calls all rules according to the simulation procedure.
__global__ void GPU_Simulation(Agent *agents, int numAgents, float conDist, float shortMov, float longMov, float width, float height) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= numAgents) return;
    
    // FALTAN REGLA 1 y 2 de implementar
    GPU_Rule1(agents, numAgents, conDist);
    GPU_Rule2(agents[i], numAgents, shortMov, longMov, width, height);

    GPU_Rule3(agents[i]);
    GPU_Rule4(agents[i]);
    GPU_Rule5(agents[i]);
}

__host__ void CPU_Simulation(Agent* agents) {
    random_device rd;
    mt19937 gen(rd());

    int cumulativeInfected = 0, prevCumulativeInfected = 0;
    int cumulativeRecovered = 0, prevCumulativeRecovered = 0;
    int cumulativeFatalities = 0, prevCumulativeFatalities = 0;

    int dayFirstInfected = -1, day50PercentInfected = -1, day100PercentInfected = -1;
    int dayFirstRecovered = -1, dayHalfRecovered = -1, dayAllRecovered = -1;
    int dayFirstFatal = -1, dayHalfFatal = -1, dayAllFatal = -1;

    agents[0].infectionStatus = 1;
    agents[0].daysInfected = 0;

    for (int day = 0; day < MAX_DAYS; ++day) {
        
        for (int mov = 0; mov < MAX_MOVEMENTS_PER_DAY; ++mov) {
            CPU_Rule1(agents, NUM_AGENTS, CONTAGION_DISTANCE);
            CPU_Rule2(agents, NUM_AGENTS);
        }

        for (int i = 0; i < NUM_AGENTS; ++i) {
            if (agents[i].infectionStatus == -2) 
                continue;
            CPU_Rule3(agents[i], gen);
            CPU_Rule4(agents[i]);
            CPU_Rule5(agents[i], gen);
        }
        cumulativeInfected = 0;
        cumulativeRecovered = 0;
        cumulativeFatalities = 0;
        for (int i = 0; i < NUM_AGENTS; ++i) {
            if (agents[i].infectionStatus != 0)
                cumulativeInfected++;
            if (agents[i].infectionStatus == 2)
                cumulativeRecovered++;
            if (agents[i].infectionStatus == -2)
                cumulativeFatalities++;
        }

        int newInfected = cumulativeInfected - prevCumulativeInfected;
        int newRecovered = cumulativeRecovered - prevCumulativeRecovered;
        int newFatalities = cumulativeFatalities - prevCumulativeFatalities;
        prevCumulativeInfected = cumulativeInfected;
        prevCumulativeRecovered = cumulativeRecovered;
        prevCumulativeFatalities = cumulativeFatalities;

        // INFECTADOS
        if (dayFirstInfected == -1 && cumulativeInfected >= 1)
            dayFirstInfected = day + 1;
        if (day50PercentInfected == -1 && cumulativeInfected >= NUM_AGENTS * 0.5)
            day50PercentInfected = day + 1;
        if (day100PercentInfected == -1 && cumulativeInfected == NUM_AGENTS)
            day100PercentInfected = day + 1;
        // RECUPERADOS
        if (dayFirstRecovered == -1 && cumulativeRecovered >= 1)
            dayFirstRecovered = day + 1;
        if (dayHalfRecovered == -1 && cumulativeRecovered >= NUM_AGENTS * 0.5)
            dayHalfRecovered = day + 1;
        if (dayAllRecovered == -1 && day == MAX_DAYS - 1) 
            dayAllRecovered = day + 1;
        //MUERTOS
        if (dayFirstFatal == -1 && cumulativeFatalities >= 1)
            dayFirstFatal = day + 1;
        if (dayHalfFatal == -1 && cumulativeFatalities >= NUM_AGENTS * 0.5)
            dayHalfFatal = day + 1;
        if (dayAllFatal == -1 && day == MAX_DAYS - 1)
            dayAllFatal = day + 1;

        // Reporte del día
        cout << "=== Dia " << day + 1 << " ===" << endl;
        cout << "Casos acumulados de contagiados: " << cumulativeInfected << endl;
        cout << "Nuevos casos de contagiados: " << newInfected << endl;
        cout << "Casos acumulados de recuperados: " << cumulativeRecovered << endl;
        cout << "Nuevos casos de recuperados: " << newRecovered << endl;
        cout << "Casos acumulados fatales: " << cumulativeFatalities << endl;
        cout << "Nuevos casos fatales: " << newFatalities << endl;
        cout << "--------------------------------------------" << endl;
    }

    // Resultados generales
    cout << "\n====== Resumen de la Simulacion (CPU) ======" << endl;
    cout << "Dia del primer contagio: " << (dayFirstInfected == -1 ? 0 : dayFirstInfected) << endl;
    cout << "Dia en que se alcanzo el 50% de contagiados: " << (day50PercentInfected == -1 ? 0 : day50PercentInfected) << endl;
    cout << "Dia en que se alcanzo el 100% de contagiados: " << (day100PercentInfected == -1 ? 0 : day100PercentInfected) << endl;
    cout << "Dia del primer recuperado: " << (dayFirstRecovered == -1 ? 0 : dayFirstRecovered) << endl;
    cout << "Dia en que se alcanzo el 50% de recuperados: " << (dayHalfRecovered == -1 ? 0 : dayHalfRecovered) << endl;
    cout << "Dia en que se alcanzo el total de recuperados: " << (dayAllRecovered == -1 ? 0 : dayAllRecovered) << endl;
    cout << "Dia del primer caso fatal: " << (dayFirstFatal == -1 ? 0 : dayFirstFatal) << endl;
    cout << "Dia en que se alcanzo el 50% de casos fatales: " << (dayHalfFatal == -1 ? 0 : dayHalfFatal) << endl;
    cout << "Dia en que se alcanzo el total de casos fatales: " << (dayAllFatal == -1 ? 0 : dayAllFatal) << endl;
}

int main() {
    Agent* CPU_Agents = new Agent[NUM_AGENTS];
    Agent* GPU_Agents = new Agent[NUM_AGENTS];
    
    // Simluacion para el CPU con tiempo
    cout << "--------------------------CPU----------------------------------\n";
    initializeAgents(CPU_Agents);
    auto startCPU = chrono::high_resolution_clock::now();
    // FUNCTION WHERE RULES ARE APPLIED
    CPU_Simulation(CPU_Agents);

    auto endCPU = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsedCPU = endCPU - startCPU;
    cout << "Sequential (CPU) simulation time: " << elapsedCPU.count() << " seconds" << endl;

    initializeAgents(GPU_Agents);
    GPU_Agents[0].infectionStatus = 1;
    Agent* d_agents;
    size_t size = NUM_AGENTS * sizeof(Agent);
    cudaMalloc((void**)&d_agents, size);
    cudaMemcpy(d_agents, GPU_Agents, size, cudaMemcpyHostToDevice);

    /*int threadsPerBlock = 256;
    int blocks = (NUM_AGENTS + threadsPerBlock - 1) / threadsPerBlock;*/

    // Simulacion para el GPU con tiempo
    cout << "--------------------------GPU----------------------------------\n";
    int cumulativeInfected = 0, prevCumulativeInfected = 0;
    int cumulativeRecovered = 0, prevCumulativeRecovered = 0;
    int cumulativeFatalities = 0, prevCumulativeFatalities = 0;

    int dayFirstInfected = -1, day50PercentInfected = -1, day100PercentInfected = -1;
    int dayFirstRecovered = -1, dayHalfRecovered = -1, dayAllRecovered = -1;
    int dayFirstFatal = -1, dayHalfFatal = -1, dayAllFatal = -1;
    auto start = chrono::high_resolution_clock::now();
    // FUNCTION WHERE RULES ARE APPLIED
    for (int day = 0; day < 30; day++) {
        GPU_Simulation << <1, 1024 >> > (d_agents, NUM_AGENTS, CONTAGION_DISTANCE, MAX_SHORT_MOVEMENT, MAX_LONG_MOVEMENT,  SIM_AREA_WIDTH, SIM_AREA_HEIGHT);
        cudaDeviceSynchronize();
        cudaMemcpy(GPU_Agents, d_agents, size, cudaMemcpyDeviceToHost);

        cumulativeInfected = 0;
        cumulativeRecovered = 0;
        cumulativeFatalities = 0;
        for (int i = 0; i < NUM_AGENTS; ++i) {
            GPU_Agents[i].maxMovmentPerDay = 10;
            if (GPU_Agents[i].infectionStatus != 0)
                cumulativeInfected++;
            if (GPU_Agents[i].infectionStatus == 2)
                cumulativeRecovered++;
            if (GPU_Agents[i].infectionStatus == -2)
                cumulativeFatalities++;
        }

        int newInfected = cumulativeInfected - prevCumulativeInfected;
        int newRecovered = cumulativeRecovered - prevCumulativeRecovered;
        int newFatalities = cumulativeFatalities - prevCumulativeFatalities;
        prevCumulativeInfected = cumulativeInfected;
        prevCumulativeRecovered = cumulativeRecovered;
        prevCumulativeFatalities = cumulativeFatalities;

        // INFECTADOS
        if (dayFirstInfected == -1 && cumulativeInfected >= 1)
            dayFirstInfected = day + 1;
        if (day50PercentInfected == -1 && cumulativeInfected >= NUM_AGENTS * 0.5)
            day50PercentInfected = day + 1;
        if (day100PercentInfected == -1 && cumulativeInfected == NUM_AGENTS)
            day100PercentInfected = day + 1;
        // RECUPERADOS
        if (dayFirstRecovered == -1 && cumulativeRecovered >= 1)
            dayFirstRecovered = day + 1;
        if (dayHalfRecovered == -1 && cumulativeRecovered >= NUM_AGENTS * 0.5)
            dayHalfRecovered = day + 1;
        if (dayAllRecovered == -1 && day == MAX_DAYS - 1)
            dayAllRecovered = day + 1;
        //MUERTOS
        if (dayFirstFatal == -1 && cumulativeFatalities >= 1)
            dayFirstFatal = day + 1;
        if (dayHalfFatal == -1 && cumulativeFatalities >= NUM_AGENTS * 0.5)
            dayHalfFatal = day + 1;
        if (dayAllFatal == -1 && day == MAX_DAYS - 1)
            dayAllFatal = day + 1;

        // Reporte del día
        cout << "=== Dia " << day + 1 << " ===" << endl;
        cout << "Casos acumulados de contagiados: " << cumulativeInfected << endl;
        cout << "Nuevos casos de contagiados: " << newInfected << endl;
        cout << "Casos acumulados de recuperados: " << cumulativeRecovered << endl;
        cout << "Nuevos casos de recuperados: " << newRecovered << endl;
        cout << "Casos acumulados fatales: " << cumulativeFatalities << endl;
        cout << "Nuevos casos fatales: " << newFatalities << endl;
        cout << "--------------------------------------------" << endl;
    }
    cudaMemcpy(GPU_Agents, d_agents, size, cudaMemcpyDeviceToHost);

    cout << "\n====== Resumen de la Simulacion (GPU) ======" << endl;
    cout << "Dia del primer contagio: " << (dayFirstInfected == -1 ? 0 : dayFirstInfected) << endl;
    cout << "Dia en que se alcanzo el 50% de contagiados: " << (day50PercentInfected == -1 ? 0 : day50PercentInfected) << endl;
    cout << "Dia en que se alcanzo el 100% de contagiados: " << (day100PercentInfected == -1 ? 0 : day100PercentInfected) << endl;
    cout << "Dia del primer recuperado: " << (dayFirstRecovered == -1 ? 0 : dayFirstRecovered) << endl;
    cout << "Dia en que se alcanzo el 50% de recuperados: " << (dayHalfRecovered == -1 ? 0 : dayHalfRecovered) << endl;
    cout << "Dia en que se alcanzo el total de recuperados: " << (dayAllRecovered == -1 ? 0 : dayAllRecovered) << endl;
    cout << "Dia del primer caso fatal: " << (dayFirstFatal == -1 ? 0 : dayFirstFatal) << endl;
    cout << "Dia en que se alcanzo el 50% de casos fatales: " << (dayHalfFatal == -1 ? 0 : dayHalfFatal) << endl;
    cout << "Dia en que se alcanzo el total de casos fatales: " << (dayAllFatal == -1 ? 0 : dayAllFatal) << endl;

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "Sequential (GPU) simulation time: " << elapsed.count() << " seconds" << endl;

    cudaFree(d_agents);

    delete[] CPU_Agents;
    delete[] GPU_Agents;
    return 0;
}