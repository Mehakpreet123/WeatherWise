from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# --------------------------------------------------------------------------------------
# 1) Config — features, targets, coco mapping
# --------------------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "temp", "dwpt", "rhum", "prcp", "wdir", "wspd", "wpgt", "pres", "coco",
    "hour", "day", "month", "weekday",
]
CATEGORICAL_FEATURES = ["city"]

TARGETS = {
    "temp": "target_temp",
    "rhum": "target_rhum",
    "pres": "target_pres",
    "wspd": "target_wspd",
}

COCO_MAP: Dict[int, str] = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
    55: "Dense drizzle", 56: "Freezing drizzle (light)", 57: "Freezing drizzle (dense)",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain", 66: "Freezing rain (light)",
    67: "Freezing rain (heavy)", 71: "Slight snow fall", 73: "Moderate snow fall",
    75: "Heavy snow fall", 77: "Snow grains", 80: "Slight rain showers",
    81: "Moderate rain showers", 82: "Violent rain showers", 85: "Slight snow showers",
    86: "Heavy snow showers", 95: "Thunderstorm", 96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail"
}

# --------------------------------------------------------------------------------------
# 2) Training utilities
# --------------------------------------------------------------------------------------

@dataclass
class ModelBundle:
    temp_model: Any
    humidity_model: Any
    pressure_model: Any
    wind_model: Any
    preprocessor: ColumnTransformer

def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

def build_pipeline() -> Pipeline:
    pre = build_preprocessor()
    return Pipeline(steps=[("pre", pre), ("model", LinearRegression())])

def train_one(df: pd.DataFrame, target_col: str) -> tuple[Pipeline, Dict[str, float]]:
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    return pipe, {"rmse": rmse, "r2": r2}

# --------------------------------------------------------------------------------------
# 3) Train all, evaluate, and save
# --------------------------------------------------------------------------------------

def train_and_save(csv_path: str, out_dir: str = ".") -> None:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=list(TARGETS.values())).copy()
    df["city"] = df["city"].astype(str)
    results = {}
    models = {}
    for name, tgt in TARGETS.items():
        model, metrics = train_one(df, tgt)
        results[name] = metrics
        models[name] = model
    print("\n=== Validation metrics (Linear Regression per target) ===")
    for name in ["temp", "rhum", "pres", "wspd"]:
        m = results[name]
        print(f"{name:>6} → RMSE: {m['rmse']:.3f}, R²: {m['r2']:.3f}")
    import pickle
    with open(os.path.join(out_dir, "temp_model.pkl"), "wb") as f:
        pickle.dump(models["temp"], f)
    with open(os.path.join(out_dir, "humidity_model.pkl"), "wb") as f:
        pickle.dump(models["rhum"], f)
    with open(os.path.join(out_dir, "pressure_model.pkl"), "wb") as f:
        pickle.dump(models["pres"], f)
    with open(os.path.join(out_dir, "wind_model.pkl"), "wb") as f:
        pickle.dump(models["wspd"], f)
    schema = {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "targets": TARGETS,
    }
    with open(os.path.join(out_dir, "feature_schema.json"), "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    print("\n✅ Saved: temp_model.pkl, humidity_model.pkl, pressure_model.pkl, wind_model.pkl, feature_schema.json")

# --------------------------------------------------------------------------------------
# 4) Inference helpers
# --------------------------------------------------------------------------------------

@dataclass
class InferenceResources:
    temp_model: Pipeline
    humidity_model: Pipeline
    pressure_model: Pipeline
    wind_model: Pipeline
    schema: Dict[str, Any]

def load_models_and_schema(model_dir: str = ".") -> InferenceResources:
    import pickle
    with open(os.path.join(model_dir, "temp_model.pkl"), "rb") as f:
        temp_model = pickle.load(f)
    with open(os.path.join(model_dir, "humidity_model.pkl"), "rb") as f:
        humidity_model = pickle.load(f)
    with open(os.path.join(model_dir, "pressure_model.pkl"), "rb") as f:
        pressure_model = pickle.load(f)
    with open(os.path.join(model_dir, "wind_model.pkl"), "rb") as f:
        wind_model = pickle.load(f)
    with open(os.path.join(model_dir, "feature_schema.json"), "r", encoding="utf-8") as f:
        schema = json.load(f)
    return InferenceResources(temp_model, humidity_model, pressure_model, wind_model, schema)

def _row_to_dataframe(schema: Dict[str, Any], row: Dict[str, Any]) -> pd.DataFrame:
    cols = schema["numeric_features"] + schema["categorical_features"]
    values = {c: [row.get(c)] for c in cols}
    return pd.DataFrame(values, columns=cols)

def predict_next_hour(resources: InferenceResources, row: Dict[str, Any]) -> Dict[str, float]:
    X = _row_to_dataframe(resources.schema, row)
    temp_pred = float(resources.temp_model.predict(X).ravel()[0])
    hum_pred  = float(resources.humidity_model.predict(X).ravel()[0])
    pres_pred = float(resources.pressure_model.predict(X).ravel()[0])
    wind_pred = float(resources.wind_model.predict(X).ravel()[0])
    return {
        "temperature": temp_pred,
        "humidity": hum_pred,
        "pressure": pres_pred,
        "wind_speed": wind_pred,
        "condition": COCO_MAP.get(int(row.get("coco", -1)), "Unknown"),
    }

# --------------------------------------------------------------------------------------
# 5) Train models inside notebook
# --------------------------------------------------------------------------------------

csv_path = "C:/Users/HP/Desktop/WeatherWise/datasets/combined.csv"
output_dir = "models"
os.makedirs(output_dir, exist_ok=True)
train_and_save(csv_path, output_dir)
