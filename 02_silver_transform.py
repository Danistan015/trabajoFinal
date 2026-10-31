# 1. Importar librerias
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.getOrCreate()

# 2. Configuracion
CATALOG = "trabajo_final_aerolineas"
BRONZE_SCHEMA = "bronze"
SILVER_SCHEMA = "silver"
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SILVER_SCHEMA}")

# 3. Cargar tablas Bronze
flights = spark.table(f"{CATALOG}.{BRONZE_SCHEMA}.flights")
airlines = spark.table(f"{CATALOG}.{BRONZE_SCHEMA}.airlines")
airports = spark.table(f"{CATALOG}.{BRONZE_SCHEMA}.airports")

# Revisar tipos de datos
for nombre, df in [("flights", flights), ("airlines", airlines), ("airports", airports)]:
    print(f"Esquema de {nombre}:")
    df.printSchema()

# 4. Preparar aeropuertos de origen y destino
origen = airports.select(
    F.col("airport_id").alias("origin_airport"),
    F.col("city").alias("origin_city"),
    F.col("country").alias("origin_country")
)

destino = airports.select(
    F.col("airport_id").alias("destination_airport"),
    F.col("city").alias("destination_city"),
    F.col("country").alias("destination_country")
)

# 5. Unir datos y calcular nuevas columnas
flights_silver = (
    flights
    .join(airlines.select("airline_id", "airline_name"), on="airline_id", how="left")
    .join(origen, on="origin_airport", how="left")
    .join(destino, on="destination_airport", how="left")
    .withColumn("delay_total", F.col("departure_delay") + F.col("arrival_delay"))
    .withColumn(
        "on_time",
        F.when(F.col("arrival_delay") <= 0, 1)
         .when(F.col("arrival_delay") > 0, 0)
    )
    .withColumn("distance_km", F.round(F.col("distance_miles") * 1.60934, 2))
    .withColumn(
        "delay_category",
        F.when(F.col("delay_total").isNull(), "Unknown")
         .when(F.col("delay_total") <= 0, "None")
         .when(F.col("delay_total") <= 15, "Short")
         .when(F.col("delay_total") <= 60, "Medium")
         .otherwise("Long")
    )
    .withColumn("flight_month", F.month("flight_date"))
    .withColumn("flight_quarter", F.quarter("flight_date"))
    .withColumn("flight_weekday", F.dayofweek("flight_date"))
    .dropDuplicates(["flight_id"])
)

# 6. Metricas por aerolinea
airline_metrics = (
    flights_silver
    .groupBy("airline_id", "airline_name")
    .agg(
        F.count("*").alias("total_flights"),
        F.round(F.avg("delay_total"), 2).alias("avg_delay"),
        F.round(F.sum("distance_km"), 2).alias("total_distance_km"),
        F.round(F.avg("on_time") * 100, 2).alias("on_time_pct")
    )
)

# 7. Metricas por aeropuerto de origen
origin_metrics = (
    flights_silver
    .groupBy("origin_airport", "origin_city", "origin_country")
    .agg(
        F.count("*").alias("departing_flights"),
        F.round(F.avg("departure_delay"), 2).alias("avg_departure_delay"),
        F.round(F.avg("on_time") * 100, 2).alias("on_time_pct")
    )
)

# 8. Metricas por aeropuerto de destino
destination_metrics = (
    flights_silver
    .groupBy("destination_airport", "destination_city", "destination_country")
    .agg(
        F.count("*").alias("arriving_flights"),
        F.round(F.avg("arrival_delay"), 2).alias("avg_arrival_delay"),
        F.round(F.avg("on_time") * 100, 2).alias("on_time_pct")
    )
)

# 9. Validar registros y valores faltantes
total_bronze = flights.count()
total_silver = flights_silver.count()

nulos_id = flights_silver.filter(F.col("flight_id").isNull() | (F.col("flight_id") == "")).count()
duplicados = (
    flights_silver.groupBy("flight_id").count()
    .filter(F.col("count") > 1).count()
)

columnas_importantes = [
    "flight_date", "airline_id", "airline_name", "origin_airport",
    "origin_city", "destination_airport", "destination_city",
    "departure_delay", "arrival_delay", "distance_miles"
]

nulos = flights_silver.select([
    F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in columnas_importantes
]).first().asDict()

print(f"Vuelos Bronze: {total_bronze} | Silver: {total_silver} | ID nulos: {nulos_id} | Duplicados: {duplicados}")
print("Valores nulos por columna:", nulos)

if nulos_id > 0 or duplicados > 0:
    raise ValueError("Se encontraron problemas en los identificadores de vuelos")

# No se inventan fechas, retrasos ni datos de aeropuertos faltantes.
# Las funciones de promedio omiten los valores nulos.

# 10. Guardar tablas Silver
tablas_silver = {
    "flights_enriched": flights_silver,
    "airline_metrics": airline_metrics,
    "origin_airport_metrics": origin_metrics,
    "destination_airport_metrics": destination_metrics
}

for nombre, df in tablas_silver.items():
    tabla = f"{CATALOG}.{SILVER_SCHEMA}.{nombre}"
    (
        df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(tabla)
    )

# 11. Verificar resultados
for nombre in tablas_silver:
    total = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.{nombre}").count()
    print(f"{nombre}: {total} registros")

print("Proceso Silver finalizado correctamente")
