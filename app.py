from flask import Flask, render_template, request
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# =========================
# START SPARK
# =========================

spark = SparkSession.builder.appName("SpamUI").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# =========================
# LOAD MODEL
# =========================

model = PipelineModel.load("hdfs://master:9000/user/gugu/spam_pipeline_model")

# =========================
# LOAD DATASET
# =========================

data = spark.read.csv("spam.csv", header=True, inferSchema=True)

data = data.select(
    col("v1").alias("label"),
    col("v2").alias("text")
)

data = data.dropna()

# =========================
# DATASET STATISTICS
# =========================

spam_count = data.filter(col("label") == "spam").count()
ham_count = data.filter(col("label") == "ham").count()

total_emails = spam_count + ham_count

spam_percent = round((spam_count / total_emails) * 100, 2)
ham_percent = round((ham_count / total_emails) * 100, 2)

# =========================
# EXAMPLE EMAILS
# =========================

spam_example = data.filter(col("label") == "spam").select("text").limit(1).collect()[0][0]
ham_example = data.filter(col("label") == "ham").select("text").limit(1).collect()[0][0]

# =========================
# WORD CLOUD + FEATURE IMPACT
# =========================

spam_text_rows = data.filter(col("label") == "spam").select("text").collect()

spam_text = " ".join(row["text"] for row in spam_text_rows)

# Create static folder if missing
if not os.path.exists("static"):
    os.makedirs("static")

# Generate word cloud
wc = WordCloud(width=800, height=400, background_color="white").generate(spam_text)

plt.figure(figsize=(10,5))
plt.imshow(wc)
plt.axis("off")
plt.tight_layout()
plt.savefig("static/wordcloud.png")

# Top spam keywords
words = spam_text.lower().split()
word_freq = Counter(words)
top_keywords = [word for word, count in word_freq.most_common(10)]

# =========================
# SPAM WORD HIGHLIGHTING
# =========================

spam_keywords = [
    "free","win","winner","offer","cash","prize","urgent",
    "claim","call","now","money","credit","loan","click"
]

def highlight_spam_words(text):

    words = text.split()
    highlighted = []

    for word in words:

        clean_word = word.lower().strip(".,!?")

        if clean_word in spam_keywords:
            highlighted.append(f"<span class='spam-word'>{word}</span>")
        else:
            highlighted.append(word)

    return " ".join(highlighted)

# =========================
# PREDICTION FUNCTION
# =========================

def predict_spam(text):

    df = spark.createDataFrame([(text,)], ["text"])

    result = model.transform(df)

    row = result.select("prediction", "probability").collect()[0]

    prediction = row["prediction"]

    spam_prob = round(row["probability"][1] * 100, 2)
    ham_prob = round(row["probability"][0] * 100, 2)

    if prediction == 1.0:
        label = "🚨 Spam Email"
    else:
        label = "✅ Not Spam"

    return label, spam_prob, ham_prob

# =========================
# ROUTE
# =========================

@app.route("/", methods=["GET", "POST"])
def home():

    result = ""
    spam_prob = ""
    ham_prob = ""
    char_count = ""
    word_count = ""
    highlighted_text = ""

    if request.method == "POST":

        message = request.form["message"]

        result, spam_prob, ham_prob = predict_spam(message)

        char_count = len(message)
        word_count = len(message.split())

        highlighted_text = highlight_spam_words(message)

    return render_template(
        "index.html",
        result=result,
        spam=spam_count,
        ham=ham_count,
        total=total_emails,
        spam_percent=spam_percent,
        ham_percent=ham_percent,
        spam_example=spam_example,
        ham_example=ham_example,
        spam_prob=spam_prob,
        ham_prob=ham_prob,
        char_count=char_count,
        word_count=word_count,
        top_keywords=top_keywords,
        highlighted_text=highlighted_text
    )

# =========================
# RUN APP
# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
