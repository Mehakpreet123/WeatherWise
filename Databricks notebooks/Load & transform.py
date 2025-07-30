# Databricks notebook source
# MAGIC %run "/weatherwise/connection"
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.sql import DataFrame

# Set base path
base_path = "abfss://weather-data-container@weatherwisestaccmehak.dfs.core.windows.net/raw/meteostat/"

# List all subfolders (each city's delta folder)
folders = [f.path for f in dbutils.fs.ls(base_path) if f.isDir()]

# Load each delta file individually
dfs = []
for path in folders:
    try:
        df = spark.read.format("delta").load(path)
        df = df.withColumn("source_path", col("_metadata.file_path"))  # if needed
        dfs.append(df)
    except Exception as e:
        print(f"Skipping {path} due to error: {e}")

# Combine all DataFrames
if dfs:
    combined_df = dfs[0]
    for df in dfs[1:]:
        combined_df = combined_df.unionByName(df)
else:
    print("No valid Delta tables found.")



# COMMAND ----------

# combined_df.tail(5)

# COMMAND ----------

combined_df.count()

# COMMAND ----------

combined_df.columns

# COMMAND ----------

combined_df.printSchema()


# COMMAND ----------

from pyspark.sql.functions import col, sum

combined_df.select([
    sum(col(c).isNull().cast("int")).alias(c + "_nulls") for c in combined_df.columns
]).show()


# COMMAND ----------

combined_df.count()

# COMMAND ----------

df_cleaned = combined_df.drop("snow", "wpgt", "tsun","source_path")


# COMMAND ----------

df_cleaned.show(4)

# COMMAND ----------

from pyspark.sql.functions import col, lit

# Step 1: Rename columns to match test data
df_cleaned = df_cleaned.withColumnRenamed("rhum", "humidity") \
                         .withColumnRenamed("pres", "pressure") \
                         .withColumnRenamed("wspd", "wind_speed") \
                         .withColumnRenamed("wpgt", "wind_gust") \
                         .withColumnRenamed("wdir", "wind_deg")

# COMMAND ----------

df_cleaned.show(5)

# COMMAND ----------

df_cleaned.count()

# COMMAND ----------

output_path = "abfss://weather-data-container@weatherwisestaccmehak.dfs.core.windows.net/curated/transformed-df"

# Save as Delta format (better for Power BI connections)
df_cleaned.write.format("delta").mode("overwrite").save(output_path)

# COMMAND ----------

# Register it as a Delta Table in Metastore
spark.sql("DROP TABLE IF EXISTS weather_data_curated_transformed")  # Optional clean-up
spark.sql(f"""
CREATE TABLE weather_data_curated_transformed
USING DELTA
LOCATION '{output_path}'
""")