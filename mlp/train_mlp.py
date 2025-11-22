from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 1. Inicializar la Sesión de Spark
spark = SparkSession.builder \
    .appName("ClasificacionTraficoMLP") \
    .getOrCreate()

# ---------------------------------------------------------
# PASO 1: Carga de Datos desde HDFS
# ---------------------------------------------------------

ruta_hdfs = "/trafico_clean"
df = spark.read.parquet(ruta_hdfs)

print(f"Total de registros cargados: {df.count()}")
df.printSchema()

# ---------------------------------------------------------
# PASO 2: Definición de Columnas
# ---------------------------------------------------------
# Define aquí cuáles columnas son tus datos de entrada y cuál es la etiqueta a predecir
columnas_features = ['Dst Port', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt', 'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts', 'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'] # Ejemplo
columna_target = 'Label' # Ejemplo: 'Malicioso', 'Normal', etc.

# Dividir datos en entrenamiento y prueba (80% training, 20% testing)
train_data, test_data = df.randomSplit([0.8, 0.2], seed=1234)

# ---------------------------------------------------------
# PASO 3: Construcción del Pipeline de Preprocesamiento
# ---------------------------------------------------------

# A. Indexar la etiqueta (Target): Convierte strings a índices numéricos (0, 1, 2...)
indexer = StringIndexer(inputCol=columna_target, outputCol="label")

# B. Ensamblar características: Crea un solo vector con todas las features
assembler = VectorAssembler(inputCols=columnas_features, outputCol="features_raw")

# C. Escalar características: Normaliza los datos (media 0, desviación std 1)
# Esto es OBLIGATORIO para que el MLP funcione bien.
scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)

# ---------------------------------------------------------
# PASO 4: Configuración del MLP
# ---------------------------------------------------------

# Necesitamos saber cuántas clases hay para la capa de salida
# y cuántas features hay para la capa de entrada.
num_clases = train_data.select(columna_target).distinct().count()
num_inputs = len(columnas_features)

# Definición de capas:
# Capa 0: Entrada (tamaño = número de features)
# Capa 1: Oculta (puedes ajustar este número, ej. 64 neuronas)
# Capa 2: Oculta (ej. 32 neuronas)
# Capa 3: Salida (tamaño = número de clases a predecir)
layers = [num_inputs, 64, 32, num_clases]

# Crear el entrenador MLP
mlp = MultilayerPerceptronClassifier(
    layers=layers,
    blockSize=128,
    seed=1234,
    maxIter=100,
    labelCol="label",
    featuresCol="features"
)

# ---------------------------------------------------------
# PASO 5: Pipeline y Entrenamiento
# ---------------------------------------------------------

# Unimos todo en un solo flujo de trabajo
pipeline = Pipeline(stages=[indexer, assembler, scaler, mlp])

print("Iniciando entrenamiento del modelo...")
model = pipeline.fit(train_data)
print("Entrenamiento finalizado.")

# ---------------------------------------------------------
# PASO 6: Evaluación
# ---------------------------------------------------------

# Hacer predicciones en el set de prueba
predictions = model.transform(test_data)

# Evaluar la precisión (Accuracy)
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", 
    predictionCol="prediction", 
    metricName="accuracy"
)

accuracy = evaluator.evaluate(predictions)
print(f"Precisión del modelo (Accuracy): {accuracy:.2f}")

# Ver algunas predicciones
predictions.select("label", "prediction", columna_target).show(10)

spark.stop()