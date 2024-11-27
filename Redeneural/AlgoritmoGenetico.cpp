#include "AlgoritmoGenetico.hpp"
#include <stdexcept>

AlgoritmoGenetico::AlgoritmoGenetico(int tamPopulacao, 
                                    int numCamadasEscondidas,
                                    int numEntradas,
                                    int numNeuroniosEscondidos,
                                    int numSaidas)
    : tamanhoPopulacao(tamPopulacao)
    , numCamadasEscondidas(numCamadasEscondidas)
    , numEntradas(numEntradas)
    , numNeuroniosEscondidos(numNeuroniosEscondidos)
    , numSaidas(numSaidas) {
    inicializarPopulacao();
}

void AlgoritmoGenetico::inicializarPopulacao() {
    populacao.clear();
    populacao.reserve(tamanhoPopulacao);
    
    for(int i = 0; i < tamanhoPopulacao; i++) {
        populacao.emplace_back(numCamadasEscondidas, 
                              numEntradas,
                              numNeuroniosEscondidos,
                              numSaidas);
    }
}

void AlgoritmoGenetico::mutacao(std::vector<double>& pesos, float taxaMutacao) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::normal_distribution<> normal_dis(0.0, 0.2);
    
    for(double& peso : pesos) {
        if(dis(gen) < taxaMutacao) {
            peso += normal_dis(gen);
        }
    }
}

void AlgoritmoGenetico::crossover(const std::vector<double>& pesos1,
                                 const std::vector<double>& pesos2,
                                 std::vector<double>& filho1,
                                 std::vector<double>& filho2) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, pesos1.size() - 1);
    
    int pontoCorte = dis(gen);
    
    filho1 = pesos1;
    filho2 = pesos2;
    
    for(int i = pontoCorte; i < pesos1.size(); i++) {
        filho1[i] = pesos2[i];
        filho2[i] = pesos1[i];
    }
}

AlgoritmoGenetico::Individuo& AlgoritmoGenetico::selecaoTorneio() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, populacao.size() - 1);
    
    Individuo& ind1 = populacao[dis(gen)];
    Individuo& ind2 = populacao[dis(gen)];
    
    return (ind1.fitness > ind2.fitness) ? ind1 : ind2;
}

void AlgoritmoGenetico::evoluir() {
    // Ordenar população por fitness (decrescente)
    std::sort(populacao.begin(), populacao.end(),
        [](const Individuo& a, const Individuo& b) {
            return a.fitness > b.fitness;
        });

    std::vector<Individuo> novaPopulacao;
    novaPopulacao.reserve(tamanhoPopulacao);

    // Elitismo: copiar os melhores indivíduos diretamente
    for(int i = 0; i < NUM_ELITISMO && i < populacao.size(); i++) {
        novaPopulacao.push_back(populacao[i]);
    }
    
    // Criar o resto da população
    while(novaPopulacao.size() < tamanhoPopulacao) {
        // Seleção com maior pressão seletiva
        Individuo& pai1 = selecaoTorneio();
        Individuo& pai2 = selecaoTorneio();
        
        std::vector<double> pesosFilho1, pesosFilho2;
        std::vector<double> pesosPai1, pesosPai2;
        
        pai1.rede.copiarCamadasParaVetor(pesosPai1);
        pai2.rede.copiarCamadasParaVetor(pesosPai2);
        
        // Crossover
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        
        if(dis(gen) < TAXA_CROSSOVER) {
            crossover(pesosPai1, pesosPai2, pesosFilho1, pesosFilho2);
        } else {
            pesosFilho1 = pesosPai1;
            pesosFilho2 = pesosPai2;
        }
        
        // Mutação adaptativa: mais mutação para indivíduos piores
        float taxaMutacaoAdaptativa = TAXA_MUTACAO * 
            (1.0f + float(novaPopulacao.size()) / tamanhoPopulacao);
        
        mutacao(pesosFilho1, taxaMutacaoAdaptativa);
        mutacao(pesosFilho2, taxaMutacaoAdaptativa);
        
        // Criar novos indivíduos
        Individuo filho1(numCamadasEscondidas, numEntradas, 
                        numNeuroniosEscondidos, numSaidas);
        Individuo filho2(numCamadasEscondidas, numEntradas, 
                        numNeuroniosEscondidos, numSaidas);
        
        filho1.rede.copiarVetorParaCamadas(pesosFilho1);
        filho2.rede.copiarVetorParaCamadas(pesosFilho2);
        
        novaPopulacao.push_back(std::move(filho1));
        if(novaPopulacao.size() < tamanhoPopulacao) {
            novaPopulacao.push_back(std::move(filho2));
        }
    }
    
    populacao = std::move(novaPopulacao);
}

void AlgoritmoGenetico::avaliarPopulacao(
    const std::function<double(RedeNeural&)>& funcaoAvaliacao) 
{
    try {
        for(auto& individuo : populacao) {
            if (funcaoAvaliacao) {
                individuo.fitness = funcaoAvaliacao(individuo.rede);
            } else {
                throw std::runtime_error("Função de avaliação inválida");
            }
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Erro ao avaliar população: " + std::string(e.what()));
    }
}

RedeNeural& AlgoritmoGenetico::getMelhorIndividuo() {
    auto it = std::max_element(populacao.begin(), populacao.end(),
        [](const Individuo& a, const Individuo& b) {
            return a.fitness < b.fitness;
        });
    return it->rede;
}

double AlgoritmoGenetico::getMelhorFitness() const {
    auto it = std::max_element(populacao.begin(), populacao.end(),
        [](const Individuo& a, const Individuo& b) {
            return a.fitness < b.fitness;
        });
    return it->fitness;
}

double AlgoritmoGenetico::getMediaFitness() const {
    double soma = 0.0;
    for(const auto& individuo : populacao) {
        soma += individuo.fitness;
    }
    return soma / populacao.size();
}

AlgoritmoGenetico::Individuo& AlgoritmoGenetico::getIndividuo(size_t index) {
    if (index >= populacao.size()) {
        throw std::out_of_range("Índice fora dos limites da população");
    }
    return populacao[index];
}

void AlgoritmoGenetico::setIndividuoFitness(size_t index, double fitness) {
    if (index >= populacao.size()) {
        throw std::out_of_range("Índice fora dos limites da população");
    }
    populacao[index].fitness = fitness;
} 