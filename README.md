# Crime Type Prediction API

API REST desenvolvida com FastAPI para prever o tipo de crime baseado em data e bairro usando modelos treinados com MLflow.

## 🚀 Como usar

### 1. Instalar dependências

```bash
cd app
pip install -r requirements.txt
```

### 2. Iniciar a API

```bash
python main.py
```

Ou com uvicorn:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

A API estará disponível em: `http://localhost:8000`

### 3. Documentação interativa

Acesse a documentação Swagger UI: `http://localhost:8000/docs`

## 📋 Endpoints

### GET `/`
Informações gerais da API

### GET `/health`
Verifica o status da API e do modelo carregado

### GET `/bairros`
Lista todos os bairros disponíveis para previsão

### GET `/predict`
Faz a previsão do tipo de crime

**Parâmetros:**
- `data` (string): Data no formato YYYY-MM-DD (ex: 2024-12-10)
- `bairro` (string): Nome do bairro (ex: Boa Viagem)

**Exemplo de requisição:**
```bash
curl "http://localhost:8000/predict?data=2024-12-10&bairro=Boa%20Viagem"
```

**Exemplo de resposta:**
```json
{
  "tipo_crime_previsto": "Homicidio/Tentativa",
  "probabilidade": 85.32,
  "data": "2024-12-10",
  "bairro": "Boa Viagem",
  "features_utilizadas": {
    "neighborhood_encoded": 0,
    "dia_semana": 1,
    "dia_mes": 10,
    "mes": 12,
    "dia_ano": 345,
    "week": 50
  }
}
```

## 🔧 Configuração

### Modelo MLflow

Por padrão, a API tenta carregar o modelo `Crime_Classification_Random_Forest` do MLflow. 

Para alterar o modelo, edite a variável `MODEL_NAME` em `main.py`:

```python
MODEL_NAME = "Crime_Classification_Random_Forest"  # ou outro modelo
```

### Mapeamento de Bairros

Os bairros disponíveis estão definidos no dicionário `NEIGHBORHOOD_MAPPING` em `main.py`. Adicione ou remova bairros conforme necessário:

```python
NEIGHBORHOOD_MAPPING = {
    "Boa Viagem": 0,
    "Piedade": 1,
    # Adicione mais bairros...
}
```

### Tipos de Crime

Os tipos de crime estão mapeados em `CRIME_TYPES`. Ajuste conforme seu modelo:

```python
CRIME_TYPES = {
    0: "Ataque a civis",
    1: "Briga",
    # ...
}
```

## 🧪 Testando a API

### Com curl:
```bash
# Health check
curl http://localhost:8000/health

# Listar bairros
curl http://localhost:8000/bairros

# Fazer previsão
curl "http://localhost:8000/predict?data=2024-12-10&bairro=Boa%20Viagem"
```

### Com Python:
```python
import requests

response = requests.get(
    "http://localhost:8000/predict",
    params={
        "data": "2024-12-10",
        "bairro": "Boa Viagem"
    }
)

print(response.json())
```

## 📦 Estrutura

```
app/
├── main.py              # Código principal da API
├── requirements.txt     # Dependências
└── README.md           # Esta documentação
```

## ⚠️ Notas

- Certifique-se de ter executado o treinamento dos modelos no notebook antes de usar a API
- Os modelos devem estar registrados no MLflow
- Ajuste os mapeamentos de bairros e crimes conforme seus dados

# Ajustes finais para a final de P6
## Melhorias do modelo
Dados que o modelo não foi apresentado para avaliação do SR2, discorreremos nesse documento sobre problemas encontrados desde o processo de pesquisa, treinamento e avaliação de suas métricas
### Coleta dos dados
Desde a fase de pesquisa houve uma complicação para encontrar uma fonte de dados que pudessem ser utilizados como base de treino, teste e validação. Foi identificada uma fonte de dados que eram disponibilizados via API da instituição Fogo Cruzado, que armazena e realiza a distribuição de diversos tipos de ocorrências registradas em sua data-base. Após a identificação da fonte, a coleta dos dados foi realizada via script Python que está disponível no repositório seguindo o caminho mlflow/main.py.
## Tratamento dos dados
Na fase de exploração e análise inicial dos dados foi perceptível alguns problemas de estruturação que seriam dores futuras, nas colunas contextInfo e victims existem dados armazenados com a estruturação de arquivos .json, como resolução foi realizado o parse de dados dados armazenados nestas determinadas colunas. 

Com o conhecimento construído da etapa anterior foi possível se observar que em nossa variável alvo, tipo de crime (obtida através dos jsons contidos em contextInfo), existia um grande desbalanceamento em relação a quantidade, com uma classe tendo mais de 10000 repetições, números muito superiores às demais, para isso foi executado um rebalanceamento dessa classe, reduzindo a apenas 10% de seus dados, para que se equipare a outras classes, levando a um modelo mais sensível e com melhores métricas
Análise de dados
Como objetivo de análise foi decidido a verificação de tipo de crime por bairro e data. Com base nisso foi possível concluir que tentativa/homicídio foi o mais predominante, com exceção de Boa Viagem.
### Preparação para o modelo
Como já definido anteriormente o modelo preditivo tomaria como base as métricas de bairro e data para retornar o tipo de crime mais provável de se acontecer, então, para o treinamento do mesmo, realizamos a normalização das variáveis necessárias, utilizando o label encoder e ajustando as datas.
Em seguida, a base de dados foi dividida em conjuntos de treinamento e teste, adotando-se a proporção de 70% para treino e 30% para teste, permitindo a avaliação adequada do desempenho do modelo em dados não vistos.
### Treinamento do modelo
Para a seleção do modelo com melhor desempenho, realizamos a comparação entre: Random Forest, Gradient Boosting, Logistic Regression, KNN e Decision Tree, utilizamos algoritmo Grid Search para otimização dos hiperparâmetros. 


O modelo de melhor desempenho foi Random Forest

Por fim, todos os modelos com seus hiperparâmetros foram armazenados no MLflow para comparações futuras
API
Para a ingestão/predição dados, foi criado uma API que recebe bairro e data e retorna o tipo de crime com maior probabilidade de ocorrência, em próximas etapas é visado a integração com um dashboard com mapas que seria disponibilizado para a força policial, de modo a tentar se organizar para obter uma melhor área de cobertura, com uma força tarefa coerente com o tipo de ocorrência predito pelo modelo. A API foi criada utilizando a tecnologia Fast API e consome o modelo diretamente do MLflow.

### Conclusão

Infelizmente, após a conclusão das etapas de treinamento e avaliação, ao se observar as métricas de desempenho, chegamos a conclusão de que o tópico trabalhado possuía variáveis latentes, que não estão representadas no conjunto de dados utilizados. Fatores socioeconômicos e comportamentais que podem impactar diretamente a dinâmica criminal, não puderam ser capturados e expressos apenas partindo das variáveis de bairro e data, dificultando a análise de padrões consistentes pelo modelo, comprometendo seu desempenho preditivo.

