#!/bin/bash
#comando para descargar archivo pesado
#aws s3 cp "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv" - --no-sign-request --region us-east-1 | hdfs dfs -put - /trafico/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv
# Configuración
BUCKET="s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms"
HDFS_DEST="/trafico"
REGION="us-east-1"

# Lista de TODOS los archivos restantes (Excluyendo Thuesday-20-02-2018)
ARCHIVOS=(
    "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv"
    "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv"
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv"
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv"
    "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv"
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"
    "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv"
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv"
)

# Crear carpeta si no existe
hdfs dfs -mkdir -p $HDFS_DEST

echo "--- Iniciando descarga masiva (~2.8 GB) ---"

for FILENAME in "${ARCHIVOS[@]}"; do
    echo "Descargando: $FILENAME ..."
    
    # Descarga Streaming: S3 -> RAM -> HDFS (Sin tocar disco local)
    aws s3 cp "$BUCKET/$FILENAME" - --no-sign-request --region $REGION | hdfs dfs -put - "$HDFS_DEST/$FILENAME"
    
    if [ $? -eq 0 ]; then
        echo "✅ [OK] Guardado en HDFS."
    else
        echo "❌ [ERROR] Falló la descarga de $FILENAME"
    fi
done

echo "--- Proceso finalizado. Verifica con: hdfs dfs -ls -h $HDFS_DEST ---"
