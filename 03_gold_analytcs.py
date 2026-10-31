# 1. Importar librerias
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder.getOrCreate()

# 2. Configuracion
CATALOG = "trabajo_final_aerolineas"
SILVER_SCHEMA = "silver"
GOLD_SCHEMA = "gold"
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{GOLD_SCHEMA}")

# 3. Cargar datos de Silver
flights = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.flights_enriched")

# 4. Retrasos por mes, dia y trimestre
monthly_delays = (
    flights
    .groupBy(F.year("flight_date").alias("year"), F.month("flight_date").alias("month"))
    .agg(
        F.count("*").alias("total_flights"),
        F.round(F.avg("arrival_delay"), 2).alias("avg_arrival_delay"),
        F.round(F.avg("departure_delay"), 2).alias("avg_departure_delay"),
        F.round(F.avg("on_time") * 100, 2).alias("on_time_pct")
    )
)

weekday_delays = (
    flights.groupBy("flight_weekday")
    .agg(
        F.count("*").alias("total_flights"),
        F.round(F.avg("arrival_delay"), 2).alias("avg_arrival_delay")
    )
)

quarterly_delays = (
    flights
    .groupBy(F.year("flight_date").alias("year"), F.col("flight_quarter").alias("quarter"))
    .agg(
        F.count("*").alias("total_flights"),
        F.round(F.avg("arrival_delay"), 2).alias("avg_arrival_delay")
    )
)

# 5. Top 5 vuelos con mayor retraso por aerolinea
# row_number da posiciones unicas y dense_rank conserva los empates.
airline_window = (
    Window.partitionBy("airline_id")
    .orderBy(F.col("delay_total").desc_nulls_last(), F.col("flight_id").asc())
)

airline_rank_window = (
    Window.partitionBy("airline_id")
    .orderBy(F.col("delay_total").desc_nulls_last())
)

top_flights = (
    flights
    .withColumn("row_number", F.row_number().over(airline_window))
    .withColumn("delay_rank", F.dense_rank().over(airline_rank_window))
    .filter(F.col("row_number") <= 5)
    .select(
        "airline_id", "airline_name", "flight_id", "flight_date",
        "origin_airport", "destination_airport", "delay_total",
        "row_number", "delay_rank"
    )
)


# 6. Ranking de aeropuertos por retraso

# Calcular el retraso promedio de los vuelos salientes

airport_delays = (
    flights
    .groupBy("origin_airport", "origin_city")
    .agg(
        F.count("*").alias("departing_flights"),
        F.round(F.avg("departure_delay"), 2).alias("avg_delay"),
        F.round(F.avg("on_time") * 100, 2).alias("on_time_pct")
    )
)

# Ordenar los aeropuertos del mayor al menor retraso

airport_window = Window.orderBy(
    F.col("avg_delay").desc_nulls_last()
)

airport_ranking = (
    airport_delays
    .withColumn(
        "airport_rank",
        F.dense_rank().over(airport_window)
    )
)


# 7. Indicadores mensuales por aerolinea
airline_monthly_kpis = (
    flights
    .groupBy(
        "airline_id", "airline_name",
        F.year("flight_date").alias("year"),
        F.month("flight_date").alias("month")
    )
    .agg(
        F.count("*").alias("total_flights"),
        F.round(F.avg("arrival_delay"), 2).alias("avg_arrival_delay"),
        F.round(F.avg("delay_total"), 2).alias("avg_total_delay"),
        F.round(F.avg("on_time") * 100, 2).alias("on_time_pct")
    )
)

# 8. Indicadores mensuales por aeropuerto
airport_monthly_kpis = (
    flights
    .groupBy(
        "origin_airport", "origin_city",
        F.year("flight_date").alias("year"),
        F.month("flight_date").alias("month")
    )
    .agg(
        F.count("*").alias("departing_flights"),
        F.round(F.avg("departure_delay"), 2).alias("avg_departure_delay"),
        F.round(F.avg("on_time") * 100, 2).alias("on_time_pct")
    )
)

# 9. Indicadores generales por aerolinea
airline_kpis = (
    flights.groupBy("airline_id", "airline_name")
    .agg(
        F.count("*").alias("total_flights"),
        F.round(F.avg("arrival_delay"), 2).alias("avg_arrival_delay"),
        F.round(F.avg("delay_total"), 2).alias("avg_total_delay"),
        F.round(F.avg("on_time") * 100, 2).alias("on_time_pct"),
        F.round(F.sum("distance_km"), 2).alias("total_distance_km")
    )
)

# 10. Ranking de puntualidad de aeropuertos
punctuality_window = Window.orderBy(F.col("on_time_pct").desc_nulls_last())
airport_punctuality = (
    airport_delays
    .withColumn("punctuality_rank", F.dense_rank().over(punctuality_window))
)

# 11. Datos por vuelo para Machine Learning
# El retraso de llegada es el objetivo, no una variable de entrada.
ml_flights = (
    flights.select(
        "flight_id", "flight_date", "airline_id", "origin_airport",
        "destination_airport", "departure_delay", "distance_km",
        "flight_month", "flight_quarter", "flight_weekday", "arrival_delay"
    )
    .filter(F.col("arrival_delay").isNotNull())
)

# 12. Guardar tablas Gold
gold_tables = {
    "monthly_delays": monthly_delays,
    "weekday_delays": weekday_delays,
    "quarterly_delays": quarterly_delays,
    "top_flights_by_airline": top_flights,
    "airport_delay_ranking": airport_ranking,
    "airline_monthly_kpis": airline_monthly_kpis,
    "airport_monthly_kpis": airport_monthly_kpis,
    "airline_kpis": airline_kpis,
    "airport_punctuality_ranking": airport_punctuality,
    "ml_flights": ml_flights
}

for nombre, df in gold_tables.items():
    tabla = f"{CATALOG}.{GOLD_SCHEMA}.{nombre}"
    (
        df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(tabla)
    )

# 13. Optimizar tablas Delta
for nombre in gold_tables:
    spark.sql(f"OPTIMIZE {CATALOG}.{GOLD_SCHEMA}.{nombre}")

# 14. Verificar resultados
for nombre in gold_tables:
    total = spark.table(f"{CATALOG}.{GOLD_SCHEMA}.{nombre}").count()
    print(f"{nombre}: {total} registros")

print("Proceso Gold finalizado correctamente")
