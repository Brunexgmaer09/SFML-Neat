#include "RedeNeural.hpp"
#include <fstream>
#include <random>

Neuronio::Neuronio(int quantidadeLigacoes) : erro(0), saida(1) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1000, 1000);
    
    pesos.resize(quantidadeLigacoes);
    for(auto& peso : pesos) {
        peso = dis(gen);
    }
}

Camada::Camada(int quantidadeNeuronios, int quantidadeLigacoes) {
    neuronios.reserve(quantidadeNeuronios);
    for(int i = 0; i < quantidadeNeuronios; i++) {
        neuronios.emplace_back(quantidadeLigacoes);
    }
}

double RedeNeural::relu(double x) {
    if(x < 0) return 0;
    return x < 10000 ? x : 10000;
}

RedeNeural::RedeNeural(int quantidadeEscondidas, 
                       int qtdNeuroniosEntrada, 
                       int qtdNeuroniosEscondida, 
                       int qtdNeuroniosSaida) :
    camadaEntrada(qtdNeuroniosEntrada + BIAS, 0),
    camadaSaida(qtdNeuroniosSaida, qtdNeuroniosEscondida + BIAS)
{
    // Inicializa camadas escondidas
    camadasEscondidas.reserve(quantidadeEscondidas);
    
    for(int i = 0; i < quantidadeEscondidas; i++) {
        int inputSize = (i == 0) ? qtdNeuroniosEntrada + BIAS : 
                                  qtdNeuroniosEscondida + BIAS;
        camadasEscondidas.emplace_back(qtdNeuroniosEscondida + BIAS, inputSize);
    }
}

void RedeNeural::calcularSaida() {
    // Calcula saídas entre a camada de entrada e primeira camada escondida
    for(int i = 0; i < camadasEscondidas[0].getQuantidadeNeuronios() - BIAS; i++) {
        double somatorio = 0;
        for(int j = 0; j < camadaEntrada.getQuantidadeNeuronios(); j++) {
            somatorio += camadaEntrada.getNeuronio(j).getSaida() * 
                        camadasEscondidas[0].getNeuronio(i).getPeso(j);
        }
        camadasEscondidas[0].getNeuronio(i).setSaida(relu(somatorio));
    }

    // Calcula saídas entre camadas escondidas
    for(size_t k = 1; k < camadasEscondidas.size(); k++) {
        for(int i = 0; i < camadasEscondidas[k].getQuantidadeNeuronios() - BIAS; i++) {
            double somatorio = 0;
            for(int j = 0; j < camadasEscondidas[k-1].getQuantidadeNeuronios(); j++) {
                somatorio += camadasEscondidas[k-1].getNeuronio(j).getSaida() * 
                            camadasEscondidas[k].getNeuronio(i).getPeso(j);
            }
            camadasEscondidas[k].getNeuronio(i).setSaida(relu(somatorio));
        }
    }

    // Calcula saídas da camada de saída
    for(int i = 0; i < camadaSaida.getQuantidadeNeuronios(); i++) {
        double somatorio = 0;
        for(int j = 0; j < camadasEscondidas.back().getQuantidadeNeuronios(); j++) {
            somatorio += camadasEscondidas.back().getNeuronio(j).getSaida() * 
                        camadaSaida.getNeuronio(i).getPeso(j);
        }
        camadaSaida.getNeuronio(i).setSaida(relu(somatorio));
    }
}

void RedeNeural::copiarParaEntrada(const std::vector<double>& vetorEntrada) {
    // Copia o vetor de entrada para a camada de entrada, exceto o BIAS
    for(int i = 0; i < camadaEntrada.getQuantidadeNeuronios() - BIAS; i++) {
        camadaEntrada.getNeuronio(i).setSaida(vetorEntrada[i]);
    }
}

void RedeNeural::copiarDaSaida(std::vector<double>& vetorSaida) {
    vetorSaida.resize(camadaSaida.getQuantidadeNeuronios());
    for(int i = 0; i < camadaSaida.getQuantidadeNeuronios(); i++) {
        vetorSaida[i] = camadaSaida.getNeuronio(i).getSaida();
    }
}

int RedeNeural::getQuantidadePesos() const {
    int soma = 0;
    
    // Soma pesos das camadas escondidas
    for(const auto& camada : camadasEscondidas) {
        for(int j = 0; j < camada.getQuantidadeNeuronios(); j++) {
            soma += camada.getNeuronio(j).getQuantidadeLigacoes();
        }
    }
    
    // Soma pesos da camada de saída
    for(int i = 0; i < camadaSaida.getQuantidadeNeuronios(); i++) {
        soma += camadaSaida.getNeuronio(i).getQuantidadeLigacoes();
    }
    
    return soma;
}

void RedeNeural::copiarVetorParaCamadas(const std::vector<double>& vetor) {
    int j = 0;

    // Copia para camadas escondidas
    for(auto& camada : camadasEscondidas) {
        for(int k = 0; k < camada.getQuantidadeNeuronios(); k++) {
            auto& neuronio = camada.getNeuronio(k);
            for(int l = 0; l < neuronio.getQuantidadeLigacoes(); l++) {
                neuronio.setPeso(l, vetor[j++]);
            }
        }
    }

    // Copia para camada de saída
    for(int k = 0; k < camadaSaida.getQuantidadeNeuronios(); k++) {
        auto& neuronio = camadaSaida.getNeuronio(k);
        for(int l = 0; l < neuronio.getQuantidadeLigacoes(); l++) {
            neuronio.setPeso(l, vetor[j++]);
        }
    }
}

void RedeNeural::copiarCamadasParaVetor(std::vector<double>& vetor) {
    vetor.clear();
    vetor.reserve(getQuantidadePesos());

    // Copia de camadas escondidas
    for(const auto& camada : camadasEscondidas) {
        for(int k = 0; k < camada.getQuantidadeNeuronios(); k++) {
            const auto& neuronio = camada.getNeuronio(k);
            for(int l = 0; l < neuronio.getQuantidadeLigacoes(); l++) {
                vetor.push_back(neuronio.getPeso(l));
            }
        }
    }

    // Copia da camada de saída
    for(int k = 0; k < camadaSaida.getQuantidadeNeuronios(); k++) {
        const auto& neuronio = camadaSaida.getNeuronio(k);
        for(int l = 0; l < neuronio.getQuantidadeLigacoes(); l++) {
            vetor.push_back(neuronio.getPeso(l));
        }
    }
}

RedeNeural RedeNeural::carregarRede(const std::string& nomeArquivo) {
    std::ifstream arquivo(nomeArquivo, std::ios::binary);
    if (!arquivo) {
        throw std::runtime_error("Não foi possível abrir o arquivo: " + nomeArquivo);
    }

    int qtdEscondida, qtdNeuroEntrada, qtdNeuroEscondida, qtdNeuroSaida;
    
    arquivo.read(reinterpret_cast<char*>(&qtdEscondida), sizeof(int));
    arquivo.read(reinterpret_cast<char*>(&qtdNeuroEntrada), sizeof(int));
    arquivo.read(reinterpret_cast<char*>(&qtdNeuroEscondida), sizeof(int));
    arquivo.read(reinterpret_cast<char*>(&qtdNeuroSaida), sizeof(int));

    RedeNeural rede(qtdEscondida, qtdNeuroEntrada, qtdNeuroEscondida, qtdNeuroSaida);

    // Lê os pesos para cada camada
    for(auto& camada : rede.camadasEscondidas) {
        for(int i = 0; i < camada.getQuantidadeNeuronios(); i++) {
            auto& neuronio = camada.getNeuronio(i);
            for(int j = 0; j < neuronio.getQuantidadeLigacoes(); j++) {
                double peso;
                arquivo.read(reinterpret_cast<char*>(&peso), sizeof(double));
                neuronio.setPeso(j, peso);
            }
        }
    }

    for(int i = 0; i < rede.camadaSaida.getQuantidadeNeuronios(); i++) {
        auto& neuronio = rede.camadaSaida.getNeuronio(i);
        for(int j = 0; j < neuronio.getQuantidadeLigacoes(); j++) {
            double peso;
            arquivo.read(reinterpret_cast<char*>(&peso), sizeof(double));
            neuronio.setPeso(j, peso);
        }
    }

    return rede;
}

void RedeNeural::salvarRede(const std::string& nomeArquivo) const {
    std::ofstream arquivo(nomeArquivo, std::ios::binary);
    if (!arquivo) {
        throw std::runtime_error("Não foi possível criar o arquivo: " + nomeArquivo);
    }

    int qtdEscondida = camadasEscondidas.size();
    int qtdNeuroEntrada = camadaEntrada.getQuantidadeNeuronios();
    int qtdNeuroEscondida = camadasEscondidas[0].getQuantidadeNeuronios();
    int qtdNeuroSaida = camadaSaida.getQuantidadeNeuronios();

    arquivo.write(reinterpret_cast<const char*>(&qtdEscondida), sizeof(int));
    arquivo.write(reinterpret_cast<const char*>(&qtdNeuroEntrada), sizeof(int));
    arquivo.write(reinterpret_cast<const char*>(&qtdNeuroEscondida), sizeof(int));
    arquivo.write(reinterpret_cast<const char*>(&qtdNeuroSaida), sizeof(int));

    // Salva os pesos de cada camada
    for(const auto& camada : camadasEscondidas) {
        for(int i = 0; i < camada.getQuantidadeNeuronios(); i++) {
            const auto& neuronio = camada.getNeuronio(i);
            for(int j = 0; j < neuronio.getQuantidadeLigacoes(); j++) {
                double peso = neuronio.getPeso(j);
                arquivo.write(reinterpret_cast<const char*>(&peso), sizeof(double));
            }
        }
    }

    for(int i = 0; i < camadaSaida.getQuantidadeNeuronios(); i++) {
        const auto& neuronio = camadaSaida.getNeuronio(i);
        for(int j = 0; j < neuronio.getQuantidadeLigacoes(); j++) {
            double peso = neuronio.getPeso(j);
            arquivo.write(reinterpret_cast<const char*>(&peso), sizeof(double));
        }
    }
}
