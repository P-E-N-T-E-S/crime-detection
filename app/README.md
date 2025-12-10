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
