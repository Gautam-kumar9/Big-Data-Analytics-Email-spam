from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

spark = SparkSession.builder.appName("SpamPrediction").getOrCreate()

model = PipelineModel.load("spam_pipeline_model")

data = spark.createDataFrame([
    ("Free money offer now",),
    ("Let's meet tomorrow",)
], ["text"])

result = model.transform(data)

result.select("text","prediction").show()

spark.stop()
