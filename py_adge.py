# ============================================================
# 0. Configuración inicial: Spark y rutas en HDFS
# ============================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, sum as Fsum, when, lit, input_file_name
)
from pyspark.sql.types import DoubleType
import matplotlib.pyplot as plt
import pandas as pd

spark = (
    SparkSession.builder
    .appName("CICIDS2018-EDA-Preprocesamiento")
    .getOrCreate()
)

# Ruta base en HDFS (ajusta si tu estructura es distinta)
RAW_PATH = "hdfs:///user/hadoop/cicids2018/raw"  # por ejemplo: /user/hadoop/cicids2018/raw/*.csv
PATTERN = RAW_PATH + "/*.csv"


# ============================================================
# 1. Cargar y validar tamaños de archivos (conteo de filas)
# ============================================================

# Leemos todos los CSV (solo para contar por archivo)
df_all = spark.read.csv(PATTERN, header=True, inferSchema=True)

df_files_counts = (
    df_all
    .withColumn("file", input_file_name())
    .groupBy("file")
    .agg(count("*").alias("n_filas"))
)

print("Conteo de filas por archivo (HDFS):")
df_files_counts.show(truncate=False)


# ============================================================
# 2. Verificar cantidad de features, tipo y orden entre archivos
#    (muestra de esquemas por archivo)
# ============================================================

# Obtenemos la lista de archivos reales desde la columna 'file'
file_paths = [row.file for row in df_files_counts.select("file").distinct().collect()]

print("\nEsquema por archivo:")
for path in file_paths:
    print(f"\nArchivo: {path}")
    df_tmp = spark.read.csv(path, header=True, inferSchema=True)
    print("Número de columnas:", len(df_tmp.columns))
    print("Columnas:", df_tmp.columns)
    print("Tipos:", df_tmp.dtypes)

# Aquí ya sabes que el archivo grande tiene columnas extra:
# ["Flow ID", "Src IP", "Dst IP", "Src Port"] y todas tienen "Timestamp"


# ============================================================
# 3. Cargar el dataset completo y conversión de tipos
#    (pasar de string/object a numérico, excepto Label)
# ============================================================

# Para el procesamiento global, volvemos a leer, sin inferSchema,
# y castear nosotros mismos a double.
df = spark.read.csv(PATTERN, header=True, inferSchema=False)

# Convertir todas las columnas excepto "Label" a double
cols_to_cast = [c for c in df.columns if c != "Label"]

for c_name in cols_to_cast:
    df = df.withColumn(c_name, col(c_name).cast(DoubleType()))

# Mantener Label como string (categórica)
df = df.withColumn("Label", col("Label").cast("string"))

print("\nEsquema después de castear a double (excepto Label):")
print(df.dtypes)


# ============================================================
# 4. Conteo global de nulos por columna (sobre todo el dataset)
# ============================================================

total_rows = df.count()
print(f"\nTotal de filas en el dataset completo: {total_rows:,}")

# Conteo de nulos por columna
exprs_nulos = [
    Fsum(col(c).isNull().cast("int")).alias(c) for c in df.columns
]
df_null_counts = df.select(exprs_nulos)

null_counts_dict = df_null_counts.collect()[0].asDict()

# Pasamos a DataFrame (pandas) para verlo ordenado
res_nan = (
    pd.Series(null_counts_dict, name="n_nulos")
    .to_frame()
    .sort_values("n_nulos", ascending=False)
)
res_nan["porcentaje"] = res_nan["n_nulos"] / total_rows * 100.0

print("\nConteo global de nulos por columna (ordenado):")
print(res_nan)


# ============================================================
# 5. Datos mayoritariamente constantes (> 99% de ceros)
#    (detección global de columnas casi constantes)
# ============================================================

# Columnas numéricas (double) que no son Label
numeric_cols = [c for c, t in df.dtypes if c != "Label" and t in ("double", "int", "bigint")]

# Conteo de ceros por columna numérica en una sola pasada
exprs_zeros = [
    Fsum((col(c) == 0).cast("int")).alias(c) for c in numeric_cols
]
df_zero_counts = df.select(exprs_zeros)
zero_counts_dict = df_zero_counts.collect()[0].asDict()

# Construimos tabla resumen en pandas
rows_const = []
for c in numeric_cols:
    n_zero = zero_counts_dict.get(c, 0)
    cero_porciento = n_zero / total_rows if total_rows > 0 else 0.0
    rows_const.append((c, n_zero, cero_porciento))

df_const = pd.DataFrame(rows_const, columns=["col", "n_ceros", "cero_porciento"])
df_const = df_const.sort_values("cero_porciento", ascending=False)

# Columnas candidatas a ser casi constantes (> 99% ceros)
casi_constantes = df_const[df_const["cero_porciento"] > 0.99]

print("\nColumnas con más del 99% de ceros:")
print(casi_constantes)

# Lista final de columnas casi constantes (las 11 que ya conoces deberían aparecer aquí)
cols_casi_const_global = [
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "Fwd Pkts/b Avg",
    "Bwd Pkts/b Avg",
    "Fwd Byts/b Avg",
    "Bwd Byts/b Avg",
    "Fwd Blk Rate Avg",
    "Bwd Blk Rate Avg",
    "Fwd URG Flags",
    "CWE Flag Count",
    "FIN Flag Cnt",
]


# ============================================================
# 6. Distribución de la variable objetivo (Label)
#    (lista + gráficas de barras)
# ============================================================

from pyspark.sql.functions import desc

df_label_dist = (
    df.groupBy("Label")
      .agg(count("*").alias("cantidad"))
      .orderBy(desc("cantidad"))
)

# Añadimos porcentaje
df_label_dist = df_label_dist.withColumn(
    "porcentaje",
    (col("cantidad") / lit(total_rows)) * 100.0
)

print("\nDistribución global de Label:")
df_label_dist.show(truncate=False)

# Opcional: gráfico de barras (recogiendo a pandas)
pdf_label_dist = df_label_dist.toPandas()

# Filtrar la clase literal 'Label' si aparece (la limpiaremos luego)
pdf_label_dist = pdf_label_dist[pdf_label_dist["Label"] != "Label"]

plt.figure(figsize=(8, 4))
plt.bar(pdf_label_dist["Label"], pdf_label_dist["cantidad"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Número de muestras")
plt.title("Distribución global de Label")
plt.tight_layout()
plt.show()


# ============================================================
# 7. Limpieza:
#    - eliminar 11 columnas casi constantes
#    - eliminar columnas extra del archivo grande + Timestamp
#    - eliminar filas con Label == 'Label'
# ============================================================

# Columnas extra del archivo grande y Timestamp
cols_id_pesado = ["Flow ID", "Src IP", "Dst IP", "Src Port", "Timestamp"]

# Eliminación de filas con Label literal 'Label'
df_clean = df.filter(col("Label") != "Label")

# Eliminación de columnas casi constantes y columnas de identificación/timestamp
drop_cols = [c for c in (cols_casi_const_global + cols_id_pesado) if c in df_clean.columns]
df_clean = df_clean.drop(*drop_cols)

print("\nColumnas eliminadas en df_clean:")
print(drop_cols)

print("\nEsquema final de df_clean (para correlación y modelado):")
print(df_clean.dtypes)

print("\nNúmero de filas tras limpiar Label literal y eliminar columnas:")
print(df_clean.count())


# ============================================================
# 8. Preparar conjunto para análisis de correlación
#    (sample opcional a pandas, si quieres hacer corr() allí)
# ============================================================

# Creamos una etiqueta binaria solo para EDA (Benign vs Attack)
df_clean = df_clean.withColumn(
    "Label_binaria",
    when(col("Label") == "Benign", lit("Benign")).otherwise(lit("Attack"))
)

# Lista de columnas numéricas (sin Label ni Label_binaria)
numeric_cols_clean = [c for c, t in df_clean.dtypes
                      if t in ("double", "int", "bigint")
                      and c not in ["Label"]]

# Muestra para correlación (por ejemplo, hasta 1e6 filas para no saturar memoria local)
df_corr_sample = df_clean.select(numeric_cols_clean + ["Label_binaria"]).sample(
    withReplacement=False, fraction=0.1, seed=42
)

# Opcional: llevar a pandas para corr()
pdf_corr_sample = df_corr_sample.toPandas()

# Matriz de correlación (pandas)
corr_matrix = pdf_corr_sample[numeric_cols_clean].corr()
print("\nMatriz de correlación calculada (pandas):")
print(corr_matrix)


# ============================================================
# 9. Estrategias de normalización y codificación de variables
#    (para usar en el pipeline de ML con Spark)
# ============================================================

from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

# Indexar la etiqueta original (multiclase) y la binaria si la necesitas
label_indexer = StringIndexer(
    inputCol="Label",
    outputCol="Label_index",
    handleInvalid="keep"
)

# (opcional) indexer binaria
label_bin_indexer = StringIndexer(
    inputCol="Label_binaria",
    outputCol="Label_binaria_index",
    handleInvalid="keep"
)

# VectorAssembler con todas las features numéricas
assembler = VectorAssembler(
    inputCols=numeric_cols_clean,
    outputCol="features_raw",
    handleInvalid="keep"
)

# Estandarización de las features
scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withMean=True,
    withStd=True
)

# Pipeline de preprocesamiento completo
preprocess_pipeline = Pipeline(stages=[
    label_indexer,
    label_bin_indexer,
    assembler,
    scaler
])

# Ajustar el pipeline sobre df_clean
preprocess_model = preprocess_pipeline.fit(df_clean)

# Aplicar transformaciones
df_prepared = preprocess_model.transform(df_clean)

print("\nEsquema de df_prepared (listo para MLlib):")
df_prepared.select("Label", "Label_index", "Label_binaria", "Label_binaria_index", "features").printSchema()
