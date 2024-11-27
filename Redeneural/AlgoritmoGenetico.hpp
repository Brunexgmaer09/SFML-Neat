#pragma once
#include "RedeNeural.hpp"
#include "FuncoesAuxiliares.hpp"
#include <vector>
#include <algorithm>
#include <random>
#include <functional>

class AlgoritmoGenetico {
public:
    struct Individuo {
        RedeNeural rede;
        double fitness;
        
        Individuo(int numCamadasEscondidas, int numEntradas, 
                 int numNeuroniosEscondidos, int numSaidas) 
            : rede(numCamadasEscondidas, numEntradas, 
                  numNeuroniosEscondidos, numSaidas), fitness(0.0) {}
    };

    AlgoritmoGenetico(int tamPopulacao, 
                      int numCamadasEscondidas,
                      int numEntradas,
                      int numNeuroniosEscondidos,
                      int numSaidas);

    void inicializarPopulacao();
    void avaliarPopulacao(const std::function<double(RedeNeural&)>& funcaoAvaliacao);
    void evoluir();
    
    // Novos métodos para acessar e modificar indivíduos
    Individuo& getIndividuo(size_t index);
    void setIndividuoFitness(size_t index, double fitness);
    size_t getTamanhoPopulacao() const { return populacao.size(); }

    RedeNeural& getMelhorIndividuo();
    double getMelhorFitness() const;
    double getMediaFitness() const;

private:
    static constexpr double TAXA_MUTACAO = 0.2;
    static constexpr double TAXA_CROSSOVER = 0.8;
    static constexpr int NUM_ELITISMO = 70;
    
    std::vector<Individuo> populacao;
    int tamanhoPopulacao;
    
    // Parâmetros da rede
    int numCamadasEscondidas;
    int numEntradas;
    int numNeuroniosEscondidos;
    int numSaidas;

    // Métodos auxiliares
    void mutacao(std::vector<double>& pesos, float taxaMutacao);
    void crossover(const std::vector<double>& pesos1, 
                  const std::vector<double>& pesos2,
                  std::vector<double>& filho1,
                  std::vector<double>& filho2);
    Individuo& selecaoTorneio();
}; 