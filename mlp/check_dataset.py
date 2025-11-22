from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# 1. Inicializar Spark
spark = SparkSession.builder \
    .appName("ExploracionTrafico") \
    .getOrCreate()

# Para que los logs no ensucien la salida, reducimos el nivel de verbosidad
spark.sparkContext.setLogLevel("WARN")

print("--- INICIANDO EXPLORACIÓN DE DATOS ---")

# 2. Cargar datos
ruta_hdfs = "/trafico_clean"
try:
    df = spark.read.parquet(ruta_hdfs)
    print(f"Datos cargados correctamente desde: {ruta_hdfs}")
except Exception as e:
    print(f"Error cargando datos: {e}")
    spark.stop()
    exit()

# 3. Ver estructura (Schema)
print("\n--- 1. ESQUEMA DETECTADO (Columnas y Tipos) ---")
# Esto te dirá si tus columnas son Double (números) o String (texto)
df.printSchema()

# 4. Ver muestra de datos reales
print("\n--- 2. MUESTRA DE DATOS (Primeras 5 filas) ---")
# truncate=False permite ver el contenido completo de las celdas
df.show(5, truncate=False)

# 5. Conteo Total
print("\n--- 3. VOLUMEN DE DATOS ---")
total_rows = df.count()
print(f"Total de registros (filas) en el dataset: {total_rows}")

# 6. Verificar Particionamiento (Columnas derivadas de carpetas)
# Si el dataset está particionado por fecha, verás una columna extra 
# (ej. 'fecha' o 'dia') que no estaba en el archivo original.
print("\n--- 4. COLUMNAS DISPONIBLES ---")
print(df.columns)

# 7. Estadísticas Básicas (Solo columnas numéricas)
# Esto ayuda a ver si hay valores extraños (ej. puertos negativos o duraciones de 0)
print("\n--- 5. ESTADÍSTICAS BÁSICAS (Min, Max, Promedio) ---")
# Seleccionamos solo tipos numéricos para que no falle el describe
numeric_cols = [f.name for f in df.schema.fields if not isinstance(f.dataType, str)]
# Si hay muchas columnas, limitamos a las primeras 5 para no saturar la pantalla
df.select(df.columns[:5]).describe().show()

print("--- FIN DEL ANÁLISIS ---")
spark.stop()