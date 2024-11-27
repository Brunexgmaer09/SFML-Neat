#pragma once
#include <vector>
#include "RedeNeural.hpp"

namespace Variaveis {
    // Constantes para a rede neural
    constexpr int POPULACAO_MAX = 1000;
    constexpr int POPULACAO_TAMANHO = 100;
    
    // Parâmetros da rede neural
    constexpr int BIRD_BRAIN_QTD_LAYERS = 1;    // Quantidade de camadas escondidas
    constexpr int BIRD_BRAIN_QTD_INPUT = 5;     // Quantidade de neurônios na entrada
    constexpr int BIRD_BRAIN_QTD_HIDE = 4;      // Quantidade de neurônios na camada escondida
    constexpr int BIRD_BRAIN_QTD_OUTPUT = 2;    // Quantidade de neurônios na saída

    // Variáveis para controle de gerações
    extern int GeracaoCompleta;
    extern std::vector<double> BestFitnessPopulacao;
    extern std::vector<double> MediaFitnessPopulacao;
    extern std::vector<double> MediaFitnessFilhos;
    
    // Variável para armazenar a melhor rede
    extern RedeNeural* MelhorRede;
}
