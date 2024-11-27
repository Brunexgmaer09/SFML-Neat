#pragma once
#include <vector>

class InputsRedeNeural {
public:
    // Métodos estáticos para processar entradas da rede
    static std::vector<double> processarEntradas(const std::vector<double>& dadosEntrada) {
        std::vector<double> entradas;
        entradas.reserve(dadosEntrada.size());
        
        // Aqui você pode adicionar qualquer pré-processamento necessário
        // dos dados de entrada antes de alimentar a rede neural
        for(const auto& dado : dadosEntrada) {
            entradas.push_back(dado);
        }
        
        return entradas;
    }
};
