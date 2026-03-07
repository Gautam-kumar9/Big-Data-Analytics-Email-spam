from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# =========================
# START SPARK
# =========================

spark = SparkSession.builder \
    .appName("SpamDetectionTraining") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# =========================
# LOAD DATASET
# =========================

data = spark.read.csv(
    "spam.csv",
    header=True,
    inferSchema=True
)

print("Dataset Loaded Successfully")

# =========================
# SELECT REQUIRED COLUMNS
# =========================

data = data.select(
    col("v1").alias("label"),
    col("v2").alias("text")
)

# =========================
# REMOVE NULL VALUES
# =========================

data = data.dropna()

# =========================
# CONVERT LABEL TO NUMERIC
# =========================

data = data.withColumn(
    "labelIndex",
    when(col("label") == "spam", 1).otherwise(0)
)

# =========================
# CLEAN TEXT
# =========================

data = data.withColumn("text", lower(col("text")))

# =========================
# FEATURE PIPELINE
# =========================

tokenizer = Tokenizer(
    inputCol="text",
    outputCol="words"
)

stopwords = StopWordsRemover(
    inputCol="words",
    outputCol="filtered"
)

hashingTF = HashingTF(
    inputCol="filtered",
    outputCol="rawFeatures",
    numFeatures=20000
)

idf = IDF(
    inputCol="rawFeatures",
    outputCol="features"
)

lr = LogisticRegression(
    featuresCol="features",
    labelCol="labelIndex",
    maxIter=20
)

pipeline = Pipeline(stages=[
    tokenizer,
    stopwords,
    hashingTF,
    idf,
    lr
])

# =========================
# TRAIN TEST SPLIT
# =========================

train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# =========================
# TRAIN MODEL
# =========================

model = pipeline.fit(train_data)

# =========================
# PREDICTIONS
# =========================

predictions = model.transform(test_data)

predictions.select("text","prediction","labelIndex").show(10, False)

# Confusion Matrix
predictions.groupBy("labelIndex","prediction").count().show()

# =========================
# EVALUATE MODEL
# =========================

evaluator = MulticlassClassificationEvaluator(
    labelCol="labelIndex",
    predictionCol="prediction",
    metricName="accuracy"
)

accuracy = evaluator.evaluate(predictions)

print("Model Accuracy:", accuracy)

# =========================
# SAVE MODEL
# =========================

model.write().overwrite().save("spam_pipeline_model")

spark.stop()
