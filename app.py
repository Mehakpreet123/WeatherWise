
from geopy.geocoders import Nominatim



import openmeteo_requests
import requests_cache
from retry_requests import retry


from flask import Flask, render_template, request, jsonify,redirect,url_for
import sqlite3
import datetime
import os
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler

from weatherwise_training_and_inference import load_models_and_schema, predict_next_hour
from utils import get_current_weather

app = Flask(__name__)

# Open-Meteo API setup
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo_client = openmeteo_requests.Client(session=retry_session)

geolocator = Nominatim(user_agent="weatherwise_app")

def get_city_coordinates(city_name):
    location = geolocator.geocode(city_name)
    if location:
        return location.latitude, location.longitude
    return None, None

from flask import Flask, render_template, request
import pandas as pd
import requests

openmeteo_url = "https://api.open-meteo.com/v1/forecast"

DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "windspeed_10m_max",
    "winddirection_10m_dominant",
    "cloudcover_mean",
    "uv_index_max"
]

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dewpoint_2m",
    "apparent_temperature",
    "windspeed_10m",
    "winddirection_10m",
    "precipitation",
    "precipitation_probability",
    "pressure_msl",
    "cloudcover",
    "uv_index"
]

# -----------------------------
# 7-day daily forecast route
# -----------------------------

@app.route("/openmeteo_forecast/", methods=["GET"])
def openmeteo_input_page():
    # Simply show input page
    return render_template("openmeteo_input.html")

@app.route("/openmeteo_forecast/<city>", methods=["GET","POST"])
def openmeteo_forecast(city=None):
    if request.method == "GET" and city is None:
        # Show input form
        return render_template("openmeteo_input.html")
    if request.method == "POST":
        city = request.form.get("city").strip()
        return redirect(url_for("openmeteo_forecast", city=city))

    if city:  # GET with city in URL
        lat, lon = get_city_coordinates(city)
        if lat is None or lon is None:
            return f"City '{city}' not found!", 404

        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": DAILY_VARIABLES,
            "timezone": "auto"
        }

        response = requests.get(openmeteo_url, params=params).json()
        daily_df = pd.DataFrame(response["daily"])
        daily_df["date"] = pd.to_datetime(daily_df["time"]).dt.date
        daily_data = daily_df.to_dict(orient="records")

        return render_template("openmeteo_daily.html", city=city, daily_data=daily_data)

    # GET without city: show input form
    return render_template("openmeteo_input.html")


# -----------------------------
# Hourly forecast for specific day
# -----------------------------
@app.route("/openmeteo_forecast/<city>/<day>", methods=["GET"])
def openmeteo_hourly(city, day):
    lat, lon = get_city_coordinates(city)

    if lat is None or lon is None:
        return f"City '{city}' not found!", 404

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": HOURLY_VARIABLES,
        "timezone": "auto"
    }

    response = requests.get(openmeteo_url, params=params).json()
    hourly_df = pd.DataFrame(response["hourly"])
    hourly_df["datetime"] = pd.to_datetime(hourly_df["time"])
    hourly_df["date"] = hourly_df["datetime"].dt.date

    day = pd.to_datetime(day).date()
    day_hours = hourly_df[hourly_df["date"] == day].to_dict(orient="records")

    return render_template(
        "openmeteo_hourly.html",
        city=city,
        day=day,
        hourly_data=day_hours
    )








# -----------------------------
# Load models & schema
# -----------------------------
MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(f"Model directory '{MODEL_DIR}' not found!")

resources = load_models_and_schema(MODEL_DIR)

# -----------------------------
# Database setup
# -----------------------------
DB_PATH = "data/predictions.db"
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    #cur.execute("DROP TABLE IF EXISTS predictions;")   # 🔄 rebuild fresh
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT,
            current_temp REAL,
            current_dwpt REAL,
            current_hum REAL,
            current_prcp REAL,
            current_wdir REAL,
            current_wspd REAL,
            current_wpgt REAL,
            current_pres REAL,
            current_coco REAL,
            pred_temp REAL,
            pred_hum REAL,
            pred_pres REAL,
            pred_wspd REAL,
            actual_temp REAL,
            actual_hum REAL,
            actual_pres REAL,
            actual_wspd REAL,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# -----------------------------
# APScheduler Setup
# -----------------------------
scheduler = BackgroundScheduler()
scheduler.start()

def update_actuals(city, record_id):
    """Fetch the actual weather after 1 hour and update DB."""
    print("function called\n")
    actual_weather = get_current_weather(city)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        UPDATE predictions
        SET actual_temp=?, actual_hum=?, actual_pres=?, actual_wspd=?
        WHERE id=?
    """, (
        actual_weather["temp"],
        actual_weather["rhum"],
        actual_weather["pres"],
        actual_weather["wspd"],
        record_id
    ))
    conn.commit()
    conn.close()
    print(f"✅ Actuals updated for record {record_id} ({city})")

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    cities = ["Dubai", "London", "Mumbai", "New York", "Tokyo"]
    return render_template("index.html", cities=cities)

@app.route("/predict", methods=["POST"])
def predict():
    city = request.form["city"]
    current_weather = get_current_weather(city)

    row = {
        "temp": current_weather.get("temp", 0),
        "dwpt": current_weather.get("dwpt", 0),
        "rhum": current_weather.get("rhum", 0),
        "prcp": current_weather.get("prcp", 0),
        "wdir": current_weather.get("wdir", 0),
        "wspd": current_weather.get("wspd", 0),
        "wpgt": current_weather.get("wpgt", 0),
        "pres": current_weather.get("pres", 0),
        "coco": current_weather.get("coco", 1),
        "hour": datetime.datetime.now().hour,
        "day": datetime.datetime.now().day,
        "month": datetime.datetime.now().month,
        "weekday": datetime.datetime.now().weekday(),
        "city": city
    }

    # Predict next-hour weather
    preds = predict_next_hour(resources, row)

    # Save in DB
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''
    INSERT INTO predictions (
        city, current_temp, current_dwpt, current_hum, current_prcp, current_wdir,
        current_wspd, current_wpgt, current_pres, current_coco,
        pred_temp, pred_hum, pred_pres, pred_wspd, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        city, row["temp"], row["dwpt"], row["rhum"], row["prcp"], row["wdir"],
        row["wspd"], row["wpgt"], row["pres"], row["coco"],
        preds["temperature"], preds["humidity"], preds["pressure"], preds["wind_speed"],
        datetime.datetime.now().isoformat(timespec="seconds")
    ))
    record_id = cur.lastrowid
    conn.commit()
    conn.close()

    # Schedule job for 1 hour later
    run_time = datetime.datetime.now() + datetime.timedelta(hours=1)
    scheduler.add_job(
        func=update_actuals,
        trigger="date",
        run_date=run_time,
        args=[city, record_id],
        id=f"update_{record_id}"
    )
    print(f"⏳ Scheduled actual fetch for record {record_id} at {run_time}")

    return jsonify({
        "city": city,
        "current_weather": row,
        "predicted": preds,
        "message": "Prediction saved and actual fetch scheduled for next hour."
    })

@app.route("/dashboard/")
def dashboard():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''
        SELECT city, current_temp, current_dwpt, current_hum, current_prcp, current_wdir,
               current_wspd, current_wpgt, current_pres, current_coco,
               pred_temp, pred_hum, pred_pres, pred_wspd,
               actual_temp, actual_hum, actual_pres, actual_wspd,
               timestamp
        FROM predictions ORDER BY timestamp DESC
    ''')
    rows = cur.fetchall()
    conn.close()

    columns = [
        "city", "current_temp", "current_dwpt", "current_hum", "current_prcp", "current_wdir",
        "current_wspd", "current_wpgt", "current_pres", "current_coco",
        "pred_temp", "pred_hum", "pred_pres", "pred_wspd",
        "actual_temp", "actual_hum", "actual_pres", "actual_wspd",
        "timestamp"
    ]
    dict_rows = [dict(zip(columns, r)) for r in rows]

    # Extract filters
    years, months, dates = set(), set(), set()
    for r in dict_rows:
        ts = datetime.datetime.fromisoformat(r["timestamp"])
        years.add(ts.year)
        months.add(ts.strftime("%B"))
        dates.add(ts.date().isoformat())

    years = sorted(list(years), reverse=True)
    months = sorted(list(months), key=lambda m: datetime.datetime.strptime(m, "%B").month)
    dates = sorted(list(dates), reverse=True)
    cities = sorted(list({r["city"] for r in dict_rows}))

    return render_template(
        "dashboard.html",
        rows=dict_rows,
        cities=cities,
        years=years,
        months=months,
        dates=dates
    )

if __name__ == "__main__":
    app.run(debug=True)
