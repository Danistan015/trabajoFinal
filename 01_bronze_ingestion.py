
# 1. Importar librerias

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.getOrCreate()


# 2. Configuracion del proyecto

CATALOG = "trabajo_final_aerolineas"
BRONZE_SCHEMA = "bronze"
VOLUME_NAME = "landing"

BASE_PATH = f"/Volumes/{CATALOG}/{BRONZE_SCHEMA}/{VOLUME_NAME}"

FLIGHTS_PATH = f"{BASE_PATH}/flights.csv"
AIRLINES_PATH = f"{BASE_PATH}/airlines.csv"
AIRPORTS_PATH = f"{BASE_PATH}/airports.csv"


# 3. Crear catalogo, esquema y volumen

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{BRONZE_SCHEMA}")
spark.sql(
    f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{BRONZE_SCHEMA}.{VOLUME_NAME}"
)


# 4. Cargar archivos CSV

flights_raw = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(FLIGHTS_PATH)
)

airlines_raw = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(AIRLINES_PATH)
)

airports_raw = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(AIRPORTS_PATH)
)


# 5. Limpiar datos de vuelos

# Ajustar tipos de datos y quitar registros incompletos

flights_bronze = (
    flights_raw
    .withColumn(
        "flight_id",
        F.trim(F.col("flight_id").cast("string"))
    )
    .withColumn(
        "flight_date",
        F.to_date(F.col("flight_date"))
    )
    .withColumn(
        "airline_id",
        F.trim(F.col("airline_id").cast("string"))
    )
    .withColumn(
        "origin_airport",
        F.trim(F.col("origin_airport").cast("string"))
    )
    .withColumn(
        "destination_airport",
        F.trim(F.col("destination_airport").cast("string"))
    )
    .withColumn(
        "departure_delay",
        F.col("departure_delay").cast("int")
    )
    .withColumn(
        "arrival_delay",
        F.col("arrival_delay").cast("int")
    )
    .withColumn(
        "distance_miles",
        F.col("distance_miles").cast("double")
    )
    .filter(
        F.col("flight_id").isNotNull()
        & (F.col("flight_id") != "")
        & F.col("airline_id").isNotNull()
        & (F.col("airline_id") != "")
        & F.col("origin_airport").isNotNull()
        & (F.col("origin_airport") != "")
    )
    .withColumn(
        "ingest_timestamp",
        F.current_timestamp()
    )
)


# 6. Limpiar datos de aerolineas

airlines_bronze = (
    airlines_raw
    .withColumn(
        "airline_id",
        F.trim(F.col("airline_id").cast("string"))
    )
    .withColumn(
        "airline_name",
        F.trim(F.col("airline_name").cast("string"))
    )
    .filter(
        F.col("airline_id").isNotNull()
        & (F.col("airline_id") != "")
    )
    .withColumn(
        "ingest_timestamp",
        F.current_timestamp()
    )
)


# 7. Limpiar datos de aeropuertos

airports_bronze = (
    airports_raw
    .withColumn(
        "airport_id",
        F.trim(F.col("airport_id").cast("string"))
    )
    .withColumn(
        "city",
        F.trim(F.col("city").cast("string"))
    )
    .withColumn(
        "country",
        F.trim(F.col("country").cast("string"))
    )
    .filter(
        F.col("airport_id").isNotNull()
        & (F.col("airport_id") != "")
    )
    .withColumn(
        "ingest_timestamp",
        F.current_timestamp()
    )
)


# 8. Validar registros

datasets = [
    ("flights", flights_raw, flights_bronze),
    ("airlines", airlines_raw, airlines_bronze),
    ("airports", airports_raw, airports_bronze)
]

for nombre, original, limpio in datasets:
    total_original = original.count()
    total_limpio = limpio.count()

    print(
        f"{nombre}: {total_original} originales, "
        f"{total_limpio} limpios, "
        f"{total_original - total_limpio} eliminados"
    )


# 9. Guardar tablas Bronze

tablas_bronze = {
    "flights": flights_bronze,
    "airlines": airlines_bronze,
    "airports": airports_bronze
}

for nombre, dataframe in tablas_bronze.items():
    tabla_destino = f"{CATALOG}.{BRONZE_SCHEMA}.{nombre}"

    (
        dataframe.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(tabla_destino)
    )


# 10. Verificar resultados

for nombre in tablas_bronze:
    tabla = spark.table(f"{CATALOG}.{BRONZE_SCHEMA}.{nombre}")
    print(f"{nombre}: {tabla.count()} registros guardados")


print("Proceso Bronze finalizado correctamente")
