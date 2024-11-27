#pragma once
#include <random>
#include <vector>
#include <algorithm>  // para std::max_element
#include <numeric>    // para std::accumulate

class FuncoesAuxiliares {
public:
    // Gerador de números aleatórios
    static double getRandomValue() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<> dis(-1000.0, 1000.0);
        return dis(gen);
    }

    // Função para calcular o melhor fitness de um conjunto de resultados
    static double calcularMelhorFitness(const std::vector<double>& resultados) {
        if(resultados.empty()) return 0.0;
        
        return *std::max_element(resultados.begin(), resultados.end());
    }

    // Função para calcular a média do fitness
    static double calcularMediaFitness(const std::vector<double>& resultados) {
        if(resultados.empty()) return 0.0;
        
        double soma = std::accumulate(resultados.begin(), resultados.end(), 0.0);
        return soma / resultados.size();
    }
};
