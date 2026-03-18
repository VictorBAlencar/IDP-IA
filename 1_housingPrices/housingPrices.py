import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# 1. CARREGAR OS DADOS
# Lemos a planilha Excel que está na mesma pasta do código
route = 'housing_prices_California (1).xlsx'
database = pd.read_excel(route)

# 2. LIMPEZA DE DADOS (Data Cleaning)
# O dataset da Califórnia tem buracos (células vazias). 
# O comando abaixo remove as linhas incompletas para o código não travar.
database = database.dropna()

# 3. SELEÇÃO DE CARACTERÍSTICAS (Features)
# Aqui escolhemos o que a IA vai olhar para decidir o preço.
# Quanto mais colunas relevantes, melhor o modelo "entende" o valor da casa.
X = database[['median_income', 'total_rooms', 'housing_median_age', 
              'latitude', 'longitude', 'population', 'total_bedrooms', 'households']]

# O 'y' é o nosso alvo (target): o que queremos prever (o preço da casa).
y = database['median_house_value']

# 4. DIVISÃO DOS DADOS
# Separamos 70% dos dados para a IA "estudar" (treino) e 30% para "testarmos" se ela aprendeu (teste).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. ESCALONAMENTO (Standardization)
# Como temos números pequenos (renda: 3.5) e números grandes (população: 2000),
# o Scaler coloca tudo numa escala parecida para a IA não se confundir.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. CRIAÇÃO E TREINAMENTO DO MODELO
# Trocamos o SVR pelo RandomForestRegressor. Ele é um "conjunto de árvores de decisão"
# e é muito mais potente e rápido para este tipo de problema.
modelo = RandomForestRegressor(n_estimators=100, random_state=42)

print("IA treinando... aguarde alguns segundos.")
modelo.fit(X_train, y_train)

# 7. PREVISÃO E AVALIAÇÃO
# Pedimos para a IA prever os preços dos 30% de dados que ela nunca viu.
y_pred = modelo.predict(X_test)

# Calculamos o RMSE (Raiz do Erro Quadrático Médio).
# Ele nos diz, em dólares, quanto a IA está errando na média.
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'\n--- RESULTADO FINAL ---')
print(f'Erro Médio do Modelo (RMSE): ${rmse:,.2f}')

# 8. COMPARATIVO REAL VS PREVISTO
# Mostramos as 5 primeiras casas para você ver como a IA chegou perto do valor real.
print("\n--- Exemplos de Previsões ---")
resultado = pd.DataFrame({'Valor Real': y_test, 'Previsão da IA': y_pred}).head(5)
print(resultado)