from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import date
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from typing import Dict

app = FastAPI(
    title="Crime Type Prediction API",
    description="API para prever o tipo de crime baseado em data e bairro",
    version="1.0.0"
)

# Carregar o modelo do MLflow (usar o melhor modelo registrado)
MODEL_NAME = "Crime_Classification_Random_Forest"  # Ajuste conforme necessário
MODEL_URI = f"models:/{MODEL_NAME}/latest"

try:
    model = mlflow.sklearn.load_model(MODEL_URI)
    print(f"✅ Modelo {MODEL_NAME} carregado com sucesso!")
except Exception as e:
    print(f"⚠️ Erro ao carregar modelo do MLflow: {e}")
    print("💡 Usando modelo local como fallback...")
    # Fallback: carregar modelo local se MLflow não estiver disponível
    model = None

# Mapeamento de bairros (ajustar conforme seus dados)
NEIGHBORHOOD_MAPPING = {
    "Boa Viagem": 0,
    "Piedade": 1,
    "Imbiribeira": 2,
    "Jardim São Paulo": 3,
    "Prado": 4,
    # Adicione mais bairros conforme necessário
}

# Mapeamento reverso de tipos de crime
CRIME_TYPES = {
    0: "Ataque a civis",
    1: "Briga",
    2: "Disparo Acidental",
    3: "Disputa",
    4: "Homicidio/Tentativa",
    5: "Sequestro/Cárcere Privado",
    6: "Tentativa/Roubo",
    7: "Tiros a esmo"
}


class PredictionRequest(BaseModel):
    data: str = Field(..., description="Data no formato YYYY-MM-DD", example="2024-12-10")
    bairro: str = Field(..., description="Nome do bairro", example="Boa Viagem")

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
            raise ValueError(f"Bairro '{bairro}' não encontrado. Bairros disponíveis: {list(NEIGHBORHOOD_MAPPING.keys())}")
        
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
