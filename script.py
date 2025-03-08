import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
import pickle

# Carrega o dataset
dados = pd.read_csv('virose.csv')  

# Exibe as primeiras linhas e informações do dataset
#print("Primeiras linhas do dataset:")
#print(dados.head())  # Mostra as primeiras 5 linhas
#print("\nInformações do dataset:")
#print(dados.info())  # Verifica tipos de dados e valores nulos

# Separa os sintomas (features) e o diagnóstico (target)
sintomas = dados.iloc[:, 0:7].values  # Todas as linhas, colunas 0 a 6 (7 sintomas)
diagnostico = dados.iloc[:, 7].values  # Todas as linhas, coluna 7 (diagnóstico)

# Cria e aplica LabelEncoder para cada sintoma
encoder_tosse = LabelEncoder()
encoder_fadiga = LabelEncoder()
encoder_dor_muscular = LabelEncoder()
encoder_febre = LabelEncoder()
encoder_diarreia = LabelEncoder()
encoder_dor_cabeca = LabelEncoder()
encoder_enjoo = LabelEncoder()

sintomas[:, 0] = encoder_tosse.fit_transform(sintomas[:, 0])  # Codifica tosse
sintomas[:, 1] = encoder_fadiga.fit_transform(sintomas[:, 1])  # Codifica fadiga
sintomas[:, 2] = encoder_dor_muscular.fit_transform(sintomas[:, 2])  # Codifica dor muscular
sintomas[:, 3] = encoder_febre.fit_transform(sintomas[:, 3])  # Codifica febre
sintomas[:, 4] = encoder_diarreia.fit_transform(sintomas[:, 4])  # Codifica diarreia
sintomas[:, 5] = encoder_dor_cabeca.fit_transform(sintomas[:, 5])  # Codifica dor de cabeça
sintomas[:, 6] = encoder_enjoo.fit_transform(sintomas[:, 6])  # Codifica enjoo

# Salva os dados codificados em um arquivo .pkl
with open('sintomas.pkl', 'wb') as f:
    pickle.dump([sintomas, diagnostico], f)

# Cria e treina o modelo Naive Bayes
modelo = GaussianNB()
modelo.fit(sintomas, diagnostico)


# Testes 
# Exemplo 1: Todos os sintomas presentes
teste1 = [[1, 1, 1, 1, 1, 1, 1]]
previsao1 = modelo.predict(teste1)
print("\nPrevisão para teste 1 (todos os sintomas presentes):", previsao1)

# Exemplo 2: Nenhum sintoma presente
teste2 = [[0, 0, 0, 0, 0, 0, 0]]
previsao2 = modelo.predict(teste2)
print("Previsão para teste 2 (nenhum sintoma presente):", previsao2)

# Exemplo 3: Apenas febre e dor de cabeça
teste3 = [[0, 0, 0, 1, 0, 1, 0]]
previsao3 = modelo.predict(teste3)
print("Previsão para teste 3 (apenas febre e dor de cabeça):", previsao3)

# Exemplo 4: Combinação aleatória de sintomas
teste4 = [[1, 0, 1, 0, 1, 0, 1]]
previsao4 = modelo.predict(teste4)
print("Previsão para teste 4 (combinação aleatória):", previsao4)
