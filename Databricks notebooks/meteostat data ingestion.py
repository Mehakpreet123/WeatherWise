# Databricks notebook source
# MAGIC %pip install meteostat

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install databricks-cli
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC acaa7090-1c86-42ee-b75a-4f1579f5212d   client/app
# MAGIC 1c7dde54-b84c-4ae2-abca-ad1c0da7a8db    ten
# MAGIC 001127e8-fb8d-4f0a-a6d8-f6745ce1be16  obj
# MAGIC Ok18Q~kyrG6dcOiCFePjd08dNETijzmQo3Ar2brA   val
# MAGIC 838e116d-453e-4a39-9104-b6c530032a04   secid
# MAGIC

# COMMAND ----------

spark.conf.set("fs.azure.account.auth.type.weatherwisestaccmehak.dfs.core.windows.net", "OAuth")
spark.conf.set("fs.azure.account.oauth.provider.type.weatherwisestaccmehak.dfs.core.windows.net", 
               "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
spark.conf.set("fs.azure.account.oauth2.client.id.weatherwisestaccmehak.dfs.core.windows.net", 
               "acaa7090-1c86-42ee-b75a-4f1579f5212d")
spark.conf.set("fs.azure.account.oauth2.client.secret.weatherwisestaccmehak.dfs.core.windows.net", 
               "Ok18Q~kyrG6dcOiCFePjd08dNETijzmQo3Ar2brA")
spark.conf.set("fs.azure.account.oauth2.client.endpoint.weatherwisestaccmehak.dfs.core.windows.net", 
               "https://login.microsoftonline.com/1c7dde54-b84c-4ae2-abca-ad1c0da7a8db/oauth2/token")




# COMMAND ----------


from meteostat import Point, Hourly
from datetime import datetime
import pandas as pd

cities = {
    "New York": Point(40.7128, -74.0060),
    "Mumbai": Point(19.0760, 72.8777),
    "London": Point(51.5074, -0.1278),
    "Dubai": Point(25.276987, 55.296249),
    "Tokyo": Point(35.6895, 139.6917)
}

start = datetime(2023, 7, 1)
end = datetime(2024, 7, 1)

for city, location in cities.items():
    print(f"📥 Fetching 1-year data for {city}...")

    data = Hourly(location, start, end)
    df = data.fetch()

    if df.empty:
        print(f"⚠️ No data available for {city}")
        continue

    df.reset_index(inplace=True)
    df["hour"] = df["time"].dt.hour
    df["day"] = df["time"].dt.day
    df["month"] = df["time"].dt.month
    df["weekday"] = df["time"].dt.weekday
    df["city"] = city

    # ✅ Convert to Spark DataFrame
    spark_df = spark.createDataFrame(df)

    # ✅ Save as Delta
    path = f"abfss://weather-data-container@weatherwisestaccmehak.dfs.core.windows.net/raw/meteostat/{city.replace(' ', '_')}"
    spark_df.write.format("delta").mode("overwrite").save(path)

    print(f"✅ Data saved in Delta format to: {path}")




# COMMAND ----------

dbutils.fs.ls("abfss://weather-data-container@weatherwisestaccmehak.dfs.core.windows.net/")
