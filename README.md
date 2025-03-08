# Diagnóstico de Viroses com Naive Bayes

Este projeto implementa um classificador de viroses utilizando o algoritmo Naive Bayes, um modelo de aprendizado de máquina supervisionado. O modelo recebe como entrada sintomas codificados e prevê o diagnóstico com base em um dataset de treinamento.

## 📂 Estrutura do Projeto
- `virose.csv` - Arquivo contendo o dataset de sintomas e diagnósticos.
- `sintomas.pkl` - Arquivo gerado contendo os dados codificados.
- `script.py` - Código responsável pelo pré-processamento dos dados, treinamento do modelo e testes.

## 📌 Dependências
Antes de executar o código, instale as bibliotecas necessárias:
```bash
pip install pandas scikit-learn
```

## 📊 Dataset
O dataset contém 7 sintomas como variáveis de entrada e uma variável alvo representando o diagnóstico. Os sintomas são:
- Tosse
- Fadiga
- Dor muscular
- Febre
- Diarreia
- Dor de cabeça
- Enjoo

Cada sintoma é codificado como 0 (ausente) ou 1 (presente).

## 🔍 Como Funciona
1. O dataset é carregado e exibido para inspeção inicial.
2. Os sintomas são separados do diagnóstico.
3. LabelEncoder é usado para converter os sintomas em valores numéricos.
4. O modelo Naive Bayes é treinado com os dados codificados.
5. O modelo é testado com diferentes combinações de sintomas.
6. Perceba que é possível, no código, mudar os testes, adicionando mais alguns, removendo ou modificando os que já existem. Teste para ver as diferentes previsões.❗📘
## 🚀 Execução
Para rodar o código, execute o script:
```bash
python3 script.py
```
O modelo irá exibir previsões para diferentes combinações de sintomas.

## 📌 Exemplo de Previsões
```bash
Previsão para teste 1 (todos os sintomas presentes): ['Alta']
Previsão para teste 2 (nenhum sintoma presente): ['Baixa']
Previsão para teste 3 (apenas febre e dor de cabeça): ['Baixa']
Previsão para teste 4 (combinação aleatória): ['Media']
```

## 📌 Melhorias Futuras
- Ajustar os hiperparâmetros do modelo para melhor precisão.
- Testar outros algoritmos de machine learning.
- Criar uma interface gráfica para facilitar o uso.

## 📄 Licença
Este projeto é de livre uso para fins acadêmicos e educacionais.

