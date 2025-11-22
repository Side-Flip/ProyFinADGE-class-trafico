from pyspark.sql.functions import col, when
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pyspark.sql.functions as F

def cargaDataset(spark):
    ruta = "hdfs://ED5:9000/trafico_clean/"
    df = spark.read.parquet(ruta)

    # Asegurar Label como string
    df = df.withColumn("Label", df["Label"].cast("string"))

    # Reemplazar NaN y null por 0 en columnas numéricas
    numeric_cols = [c for c, t in df.dtypes if t in ("double", "float", "int", "bigint")]
    df = df.fillna(0, subset=numeric_cols)

    # Convertir NaN que fillna no toca
    df = df.select([
        col(c).cast("double").alias(c) if c in numeric_cols else col(c)
        for c in df.columns
    ])

    df = df.select([
        when(
            col(c).isNull() |
            F.isnan(col(c)) |
            (col(c) == float("inf")) |
            (col(c) == float("-inf")),
            0
        ).otherwise(col(c)).alias(c)
        if c in numeric_cols else col(c)
        for c in df.columns
    ])

    df.printSchema()

   # df = df.limit(100)

    return df


def randomForest():
    # Sesión Spark
    spark = SparkSession.builder \
        .appName("RF_CSE_CIC_IDS2018") \
        .getOrCreate()

    # Cargar dataset
    df = cargaDataset(spark)

    # Features: todas excepto Label
    feature_cols = [c for c in df.columns if c != "Label"]

    # Convertir etiqueta a índice numérico
    label_indexer = StringIndexer(
        inputCol="Label",
        outputCol="label_index"
    )

    # Vector de características
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features_vector"
    )

    # Escalar
    scaler = StandardScaler(
        inputCol="features_vector",
        outputCol="features_scaled"
    )

    # Random Forest
    rf = RandomForestClassifier(
        labelCol="label_index",
        featuresCol="features_scaled",
        numTrees=100,
        maxDepth=10
    )

    # Pipeline
    pipeline = Pipeline(stages=[label_indexer, assembler, scaler, rf])

    # División Train/Test
    train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)

    # Entrenar modelo
    model = pipeline.fit(train_df)

    # Predicciones
    preds = model.transform(test_df)
    preds.select("Label", "label_index", "prediction").show(10)

    # Evaluación
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label_index",
        predictionCol="prediction"
    )

    print("Accuracy:", evaluator.evaluate(preds, {evaluator.metricName: "accuracy"}))
    print("Precision:", evaluator.evaluate(preds, {evaluator.metricName: "weightedPrecision"}))
    print("Recall:", evaluator.evaluate(preds, {evaluator.metricName: "weightedRecall"}))
    print("F1 Score:", evaluator.evaluate(preds, {evaluator.metricName: "f1"}))

    spark.stop()


if __name__ == "__main__":
    randomForest()

