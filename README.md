# WeatherWise
WeatherWise is a data engineering and machine learning pipeline that delivers real-time, next-hour temperature predictions by integrating historical and live weather data sources.  The project automates data ingestion, feature engineering, model training, and real-time inference, using a cloud-native architecture.  
# Sample workflow:  
       ┌────────────────────┐          │  Historical Data   │ 
	Meteostat (Local & Cloud)
          └────────────────────┘
                    ↓
     ┌─────────────────────────────┐
     │  Local Training (Baseline)  │
     │  - LightGBM / XGBoost       │
     │  - Preprocessing / EDA       │
     └─────────────────────────────┘
                    ↓
     Save working notebook, features, model config
                    ↓
     ┌─────────────────────────────┐
     │     Databricks (Scaling)    │
     │  - Load larger dataset      │
     │  - Distributed training      │
     │  - MLflow model tracking │
     └─────────────────────────────┘
                    ↓
     ┌─────────────────────────────┐
     │ Azure ML / Functions / Web │
     │ Real-time prediction + API │
     └─────────────────────────────┘
<img width="479" height="628" alt="image" src="https://github.com/user-attachments/assets/0b6fdef1-742b-4e0b-867c-dc94373e0956" />

