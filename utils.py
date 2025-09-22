import requests
import numpy as np

API_KEY = "77fb2902de1cef8c5c9e3b1531b723a7"

# Approximate dew point using temp & humidity
def calculate_dew_point(temp, rh):
    a = 17.27
    b = 237.7
    alpha = ((a * temp) / (b + temp)) + np.log(rh / 100.0)
    dew_point = (b * alpha) / (a - alpha)
    return dew_point

def get_current_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    res = requests.get(url).json()

    temp = res["main"]["temp"]
    humidity = res["main"]["humidity"]
    pressure = res["main"]["pressure"]

    wind = res.get("wind", {})
    wind_speed = wind.get("speed", 0)
    gust = wind.get("gust", 0)
    wdir = wind.get("deg", 0)

    prcp = res.get("rain", {}).get("1h", 0)  # rainfall in last hour
    dwpt = calculate_dew_point(temp, humidity)

    # Map OpenWeatherMap weather ID to COCO_MAP integer (simplified)
    weather_id = res["weather"][0]["id"]
    if 200 <= weather_id < 300:
        coco = 95  # thunderstorm
    elif 300 <= weather_id < 400:
        coco = 51  # drizzle
    elif 500 <= weather_id < 600:
        coco = 61  # rain
    elif 600 <= weather_id < 700:
        coco = 71  # snow
    elif 700 <= weather_id < 800:
        coco = 45  # fog/mist
    elif weather_id == 800:
        coco = 0   # clear sky
    elif 801 <= weather_id < 900:
        coco = 3   # cloudy
    else:
        coco = 0   # unknown

    return {
    "temp": temp,
    "dwpt": dwpt,
    "rhum": humidity,
    "prcp": prcp,
    "wdir": wdir,
    "wspd": wind_speed,
    "wpgt": gust,
    "pres": pressure,
    "coco": coco
}


# Example usage:
# weather = get_current_weather("Dubai")
# print(weather)
