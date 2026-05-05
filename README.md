# 📧 Big Data Analytics – Email Spam Detection (Hadoop + Spark + Flask)

This project builds an **Email Spam Detection System using Big Data technologies**.
It uses **Apache Hadoop (HDFS)** for storage, **Apache Spark MLlib** for training the spam classification model, and **Flask** to create an interactive web dashboard where users can test emails and analyze dataset insights.

The system classifies emails as **Spam or Ham (Not Spam)** and provides **visual analytics such as charts, word clouds, and feature explanations**.

---

# 🚀 Project Features

### Spam Detection

* Classifies emails into **Spam or Not Spam**
* Uses **Spark Machine Learning (Logistic Regression)**

### Interactive Dashboard

* Email input box for prediction
* Spam / Ham prediction result
* Probability scores

### Email Analysis

* Word count
* Character count

### Dataset Analytics

* Total emails
* Spam count
* Ham count
* Spam percentage

### Data Visualization

* Spam vs Ham **Pie Chart**
* **Word Cloud of spam emails**

### Explainable AI

* **Top spam keywords (feature impact)**
* **Highlighted spam words in user input**

---

# 🏗 System Architecture

```
Dataset (spam.csv)
        │
        ▼
   Hadoop HDFS Storage
        │
        ▼
   Apache Spark ML Training
        │
        ▼
Saved Model (PipelineModel)
        │
        ▼
      Flask Web App
        │
        ▼
 Interactive Dashboard + Spam Prediction
```

---

# 📂 Project Structure

```
Big-Data-Analytics-Email-spam
│
├── spam_ui
│   ├── app.py
│   ├── train_model.py
│   │
│   ├── templates
│   │   └── index.html
│   │
│   ├── static
│   │   └── wordcloud.png
│
├── spam.csv
├── README.md
```

---

# ⚙️ System Requirements

* Ubuntu Linux
* Python 3.8+
* Java 11
* Hadoop 3.x
* Apache Spark 3.5+
* Git

---

# 1️⃣ Install Java

```
sudo apt update
sudo apt install openjdk-11-jdk -y
```

Verify installation:

```
java -version
```

---

# 2️⃣ Install Hadoop

Download Hadoop:

```
wget https://downloads.apache.org/hadoop/common/hadoop-3.3.6/hadoop-3.3.6.tar.gz
```

Extract:

```
tar -xvzf hadoop-3.3.6.tar.gz
sudo mv hadoop-3.3.6 /opt/hadoop
```

---

# 3️⃣ Configure Hadoop Environment

Edit `.bashrc`

```
nano ~/.bashrc
```

Add:

```
export HADOOP_HOME=/opt/hadoop
export PATH=$PATH:$HADOOP_HOME/bin
export PATH=$PATH:$HADOOP_HOME/sbin
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
```

Reload:

```
source ~/.bashrc
```

---

# 4️⃣ Configure HDFS

Create storage directories:

```
mkdir -p ~/hdfs/namenode
mkdir -p ~/hdfs/datanode
```

Edit:

```
/opt/hadoop/etc/hadoop/core-site.xml
```

Add:

```
<configuration>
 <property>
  <name>fs.defaultFS</name>
  <value>hdfs://master:9000</value>
 </property>
</configuration>
```

Edit:

```
hdfs-site.xml
```

Add:

```
<property>
 <name>dfs.replication</name>
 <value>1</value>
</property>
```

---

# 5️⃣ Start Hadoop

Format NameNode:

```
hdfs namenode -format
```

Start HDFS:

```
start-dfs.sh
```

Verify:

```
jps
```

Expected output:

```
NameNode
DataNode
SecondaryNameNode
```

---

# 6️⃣ Install Apache Spark

Download Spark:

```
wget https://downloads.apache.org/spark/spark-3.5.1/spark-3.5.1-bin-hadoop3.tgz
```

Extract:

```
tar -xvzf spark-3.5.1-bin-hadoop3.tgz
sudo mv spark-3.5.1-bin-hadoop3 /opt/spark
```

---

# 7️⃣ Configure Spark

Edit `.bashrc`

```
export SPARK_HOME=/opt/spark
export PATH=$PATH:$SPARK_HOME/bin
```

Reload:

```
source ~/.bashrc
```

---

# 8️⃣ Start Spark Cluster

```
cd /opt/spark
sbin/start-master.sh
sbin/start-worker.sh spark://master:7077
```

Verify:

```
jps
```

Expected:

```
Master
Worker
```

Spark UI:

```
http://master:8080
```

---

# 9️⃣ Install Python Libraries

```
pip install pyspark flask pandas wordcloud matplotlib
```

---

# 🔟 Upload Dataset to HDFS

```
hdfs dfs -mkdir -p /user/gugu
hdfs dfs -put spam.csv /user/gugu/
```

Verify:

```
hdfs dfs -ls /user/gugu
```

---

# 🤖 Train Spam Detection Model

```
spark-submit train_model.py
```

Training steps:

1. Load dataset
2. Text preprocessing
3. Tokenization
4. Stopword removal
5. TF-IDF feature extraction
6. Logistic Regression training
7. Save pipeline model

Model saved to:

```
hdfs://master:9000/user/gugu/spam_pipeline_model
```

---

# 🌐 Run the Web Application

```
cd spam_ui
spark-submit app.py
```

Open:

```
http://localhost:5000
```

---

# 📊 Dataset

This project uses a **trimmed and preprocessed version of the Enron Email Dataset**.

### Dataset Statistics

| Metric          | Value |
| --------------- | ----- |
| Total Emails    | 5572  |
| Spam Emails     | 747   |
| Ham Emails      | 4825  |
| Spam Percentage | 13.4% |
| Ham Percentage  | 86.6% |

Example:

```
ham, Hey are we meeting tomorrow?
spam, Congratulations! You won a free prize
```

---

# 📊 Model Performance & Results

### Accuracy Comparison

| Model               | Accuracy     |
| ------------------- | ------------ |
| Logistic Regression | **97.53%** ✅ |
| Naive Bayes         | 95.80%       |
| Random Forest       | 96.10%       |
| Decision Tree       | 93.40%       |

### 🏆 Best Model: Logistic Regression

Logistic Regression achieved the highest accuracy of **97.53%** due to its ability to handle high-dimensional TF-IDF features effectively.

---

# 📊 Confusion Matrix

```
             Predicted
           Ham     Spam
Actual Ham 4718    107
Actual Spam  40    707
```

### Explanation

* True Positives → 707
* True Negatives → 4718
* False Positives → 107
* False Negatives → 40

---

# 📊 Dataset Visualization

### Spam vs Ham Distribution

![Spam vs Ham](spam_ham_distribution.png)

### Spam vs Ham Pie Chart

![Pie Chart](spam_ham_pie.png)

### Email Length Distribution

![Length Distribution](email_length_distribution.png)

### Top Spam Words

![Top Words](top_spam_words.png)

### Confusion Matrix

![Confusion Matrix](confusion_matrix_updated.png)

---

# 📈 Example Prediction

Input:

```
Free money offer waiting for you
```

Output:

```
🚨 Spam Email
Spam Probability: 92%
Ham Probability: 8%
```

Highlighted:

```
[Free] [money] [offer]
```

---

# 🧠 Machine Learning Model

Algorithm:

```
Logistic Regression
```

Pipeline:

```
Tokenizer
StopWordsRemover
HashingTF
IDF
LogisticRegression
```

---

# 🔮 Future Improvements

* Advanced confusion matrix analysis
* Real-time email integration
* Advanced NLP preprocessing
* Deep learning spam detection

---

# 👨‍💻 Author

**Gautam Kumar**
and
**Saahil Kapoor**

GitHub:
https://github.com/Gautam-kumar9

---

# ⭐ Support

If you like this project, give it a ⭐ on GitHub.
