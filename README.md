# 🧠 Sentiment Analysis with Google BigQuery and BERT

This project showcases a full end-to-end machine learning pipeline for sentiment analysis using tweets, enhanced with cloud-based data handling via **Google BigQuery** and transformer-based sentiment prediction using **Hugging Face Transformers**.

It demonstrates scalable data analysis and model evaluation, all integrated into a cloud-native workflow ideal for modern ML applications.

---

## 🚀 Project Highlights

- ✅ Upload real-world tweet data to **Google BigQuery**
- ✅ Analyze sentiment label distributions and trends using **SQL**
- ✅ Use a **BERT-based model** (`cardiffnlp/twitter-roberta-base-sentiment`) to classify tweets
- ✅ Upload model predictions back to BigQuery
- ✅ Visualize model accuracy with a **confusion matrix**

---

## 🗃️ Dataset Columns

| Column             | Description                                            |
|--------------------|--------------------------------------------------------|
| `id`               | Unique tweet identifier                                |
| `text`             | Tweet content                                          |
| `label`            | True sentiment label (0 = negative, 1 = neutral, 2 = positive) |
| `model_label_text` | Text prediction by model (e.g., Neutral)               |
| `model_label`      | Mapped prediction as an integer (0, 1, 2)              |

---


## 📦 Tools Used

- Google BigQuery
- Hugging Face Transformers
- CardiffNLP RoBERTa Sentiment Model
- Python (pandas, seaborn, matplotlib)
- Google Colab / Jupyter Notebooks

---

## 📤 Step 1: Upload Data to BigQuery

Upload your preprocessed sentiment analysis dataset using `pandas_gbq.to_gbq()`:

```python
from pandas_gbq import to_gbq
to_gbq(df, "tweets_sa.tweets", project_id="sentiment-analysis-09", if_exists="replace")
```

---

## 🧠 Step 2: Run Model Predictions

Use a pretrained BERT model to predict tweet sentiment:

```python
from transformers import pipeline
model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
predictions = model(df["text"].tolist())
df["model_label_text"] = [p["label"] for p in predictions]
df["model_label"] = df["model_label_text"].map({"Neg": 0, "Neutral": 1, "Pos": 2})
```

Then upload predictions to BigQuery:

```python
to_gbq(df, "tweets_sa.tweets_with_predictions", project_id="sentiment-analysis-09", if_exists="replace")
```

---

## 📊 Step 3: Evaluate with Confusion Matrix

Query BigQuery to compare true vs predicted labels:

```sql
SELECT label AS true_label, model_label AS predicted_label, COUNT(*) AS count
FROM `sentiment-analysis-09.tweets_sa.tweets_with_predictions`
GROUP BY true_label, model_label
```

Visualize the results:

```python
import seaborn as sns
import matplotlib.pyplot as plt

conf_matrix = df_confusion.pivot(index='true_label', columns='predicted_label', values='count').fillna(0)
conf_matrix = conf_matrix.astype(int)

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix: True vs Predicted Sentiment Labels")
```

---

## 🧠 What I Learned

- Integrating BigQuery with real ML workflows
- Leveraging pretrained BERT models for text classification
- Running SQL-based evaluations and building dashboards
- Deploying cloud-ready pipelines suitable for scalable ML

