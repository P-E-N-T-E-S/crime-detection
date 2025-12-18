from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import date
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import json
import os
from typing import Dict

app = FastAPI(
    title="Crime Type Prediction API",
    description="API para prever o tipo de crime baseado em data e bairro",
    version="1.0.0"
)

# Carregar o modelo do MLflow (usar o melhor modelo registrado)
MODEL_NAME = "Crime_Classification_Random_Forest"  # Ajuste conforme necessário
MODEL_URI = f"models:/{MODEL_NAME}/latest"


def find_local_mlflow_model(root_search: str = None):
    """
    Procura por modelos salvos localmente na pasta `mlruns/**/models/*/artifacts`.
    Retorna o primeiro modelo carregável (mais recente por modificação do arquivo `MLmodel`).
    """
    if root_search is None:
        root_search = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..'))

    candidates = []
    for dirpath, dirs, files in os.walk(root_search):
        if 'MLmodel' in files and os.path.basename(dirpath) == 'artifacts':
            candidates.append(os.path.join(dirpath))

    if not candidates:
        return None

    # ordenar por data de modificação do arquivo MLmodel (mais recente primeiro)
    candidates.sort(key=lambda p: os.path.getmtime(
        os.path.join(p, 'MLmodel')), reverse=True)

    for cand in candidates:
        try:
            m = mlflow.sklearn.load_model(cand)
            print(f"✅ Modelo local carregado de: {cand}")
            return m
        except Exception as e:
            print(f"⚠️ Falha ao carregar candidato {cand}: {e}")
            continue

    return None


# Tentar carregar primeiro do Model Registry (se houver), senão procurar localmente em mlruns
model = None
try:
    model = mlflow.sklearn.load_model(MODEL_URI)
    print(f"✅ Modelo {MODEL_NAME} carregado do Model Registry com sucesso!")
except Exception as e:
    print(f"⚠️ Erro ao carregar modelo do Model Registry: {e}")
    print("🔎 Tentando localizar modelos locais em pastas 'mlruns'...")
    model = find_local_mlflow_model()
    if model is None:
        print("⚠️ Nenhum modelo local encontrado em 'mlruns'. Use o notebook para treinar/logar um modelo.")

# Carregar mapeamento de bairros do arquivo JSON


def load_neighborhood_mapping():
    try:
        json_path = os.path.join(os.path.dirname(
            __file__), 'neighborhood_mapping.json')
        with open(json_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        print(f"✅ Mapeamento de bairros carregado: {len(mapping)} bairros")
        return mapping
    except FileNotFoundError:
        print("⚠️ Arquivo neighborhood_mapping.json não encontrado. Execute o notebook primeiro!")
        return {}
    except Exception as e:
        print(f"⚠️ Erro ao carregar mapeamento de bairros: {e}")
        return {}

# Carregar mapeamento de tipos de crime do arquivo JSON


def load_crime_type_mapping():
    try:
        json_path = os.path.join(os.path.dirname(
            __file__), 'crime_type_mapping.json')
        with open(json_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        # Inverter o mapeamento: código -> nome
        inverted = {v: k for k, v in mapping.items()}
        print(
            f"✅ Mapeamento de tipos de crime carregado: {len(inverted)} tipos")
        return inverted
    except FileNotFoundError:
        print("⚠️ Arquivo crime_type_mapping.json não encontrado. Execute o notebook primeiro!")
        return {}
    except Exception as e:
        print(f"⚠️ Erro ao carregar mapeamento de crimes: {e}")
        return {}


# Carregar mapeamentos
NEIGHBORHOOD_MAPPING = load_neighborhood_mapping()
CRIME_TYPES = load_crime_type_mapping()


class PredictionRequest(BaseModel):
    data: str = Field(..., description="Data no formato YYYY-MM-DD",
                      example="2024-12-10")
    bairro: str = Field(..., description="Nome do bairro",
                        example="Boa Viagem")

    class Config:
        json_schema_extra = {
            "example": {
                "data": "2024-12-10",
                "bairro": "Boa Viagem"
            }
        }


class PredictionResponse(BaseModel):
    tipo_crime_previsto: str
    probabilidade: float
    data: str
    bairro: str
    features_utilizadas: Dict


def preparar_features(data_str: str, bairro: str) -> pd.DataFrame:
    """
    Prepara as features para o modelo a partir da data e bairro
    """
    try:
        # Converter string para datetime
        data_obj = pd.to_datetime(data_str)

        # Verificar se o bairro existe no mapeamento
        if bairro not in NEIGHBORHOOD_MAPPING:
            raise ValueError(
                f"Bairro '{bairro}' não encontrado. Bairros disponíveis: {list(NEIGHBORHOOD_MAPPING.keys())}")

        # Extrair features temporais
        dia_semana = data_obj.dayofweek
        dia_mes = data_obj.day
        mes = data_obj.month
        dia_ano = data_obj.dayofyear
        week = data_obj.isocalendar().week
        neighborhood_encoded = NEIGHBORHOOD_MAPPING[bairro]

        # Criar DataFrame com as features na ordem correta
        features = pd.DataFrame({
            'neighborhood_encoded': [neighborhood_encoded],
            'dia_semana': [dia_semana],
            'dia_mes': [dia_mes],
            'mes': [mes],
            'dia_ano': [dia_ano],
            'week': [week]
        })

        return features

    except Exception as e:
        raise ValueError(f"Erro ao preparar features: {str(e)}")


@app.get("/")
def root():
    """
    Endpoint raiz com informações da API
    """
    return {
        "message": "Crime Type Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict?data=YYYY-MM-DD&bairro=NomeDoBairro",
            "health": "/health",
            "bairros": "/bairros"
        }
    }


@app.get("/health")
def health_check():
    """
    Verifica o status da API e do modelo
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": MODEL_NAME
    }


@app.get("/bairros")
def listar_bairros():
    """
    Lista todos os bairros disponíveis para previsão
    """
    return {
        "bairros_disponiveis": list(NEIGHBORHOOD_MAPPING.keys()),
        "total": len(NEIGHBORHOOD_MAPPING)
    }


@app.get("/predict", response_model=PredictionResponse)
def predict_crime_type(data: str, bairro: str):
    """
    Prediz o tipo de crime baseado na data e bairro

    Args:
        data: Data no formato YYYY-MM-DD (ex: 2024-12-10)
        bairro: Nome do bairro (ex: Boa Viagem)

    Returns:
        PredictionResponse com o tipo de crime previsto e probabilidade
    """
    try:
        # Verificar se o modelo está carregado
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Modelo não disponível. Execute o treinamento no notebook primeiro."
            )

        # Preparar features
        features = preparar_features(data, bairro)

        # Fazer previsão
        predicao = model.predict(features)[0]

        # Obter probabilidades (se o modelo suportar)
        if hasattr(model, 'predict_proba'):
            probabilidades = model.predict_proba(features)[0]
            probabilidade_maxima = float(probabilidades[predicao])
        else:
            probabilidade_maxima = 1.0

        # Obter nome do tipo de crime
        tipo_crime = CRIME_TYPES.get(predicao, f"Desconhecido ({predicao})")

        return PredictionResponse(
            tipo_crime_previsto=tipo_crime,
            probabilidade=round(probabilidade_maxima * 100, 2),
            data=data,
            bairro=bairro,
            features_utilizadas={
                "neighborhood_encoded": int(features['neighborhood_encoded'].values[0]),
                "dia_semana": int(features['dia_semana'].values[0]),
                "dia_mes": int(features['dia_mes'].values[0]),
                "mes": int(features['mes'].values[0]),
                "dia_ano": int(features['dia_ano'].values[0]),
                "week": int(features['week'].values[0])
            }
        )

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao fazer previsão: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
