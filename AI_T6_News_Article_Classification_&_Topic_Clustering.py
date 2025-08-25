# ------------------------------------------------------------
# AI Task 6 - News Article Classification & Topic Clustering
# ------------------------------------------------------------
# ------------------------------------------------------------

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.decomposition import LatentDirichletAllocation

# ------------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------------
print("📂 Loading dataset...")
df = pd.read_json("c:/venv/data/News_Category_Dataset_v3.json", lines=True)

# Combine headline + short_description to create full text
df["text"] = df["headline"] + " " + df["short_description"]

print(f"✅ Dataset loaded with {len(df)} samples and {df['category'].nunique()} categories.")

# ------------------------------------------------------------
# Text Preprocessing Function
# ------------------------------------------------------------
def clean_text(text):
    """Lowercase, remove punctuation/digits, remove extra spaces"""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)  # keep only alphabets and spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["text"].apply(clean_text)

# ------------------------------------------------------------
# 2. Supervised Learning: News Classification
# ------------------------------------------------------------
print("\n🔹 Training Supervised Model...")

X = df["clean_text"]
y = df["category"]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorizer for classification
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Logistic Regression Classifier
clf = LogisticRegression(max_iter=200)
clf.fit(X_train_tfidf, y_train)

# Evaluate model
y_pred = clf.predict(X_test_tfidf)
print("\n📊 Classification Report (Supervised):")
print(classification_report(y_test, y_pred))

# ------------------------------------------------------------
# 3. Unsupervised Learning: Topic Modeling with LDA
# ------------------------------------------------------------
print("\n🔹 Training Unsupervised Model with LDA (20 Topics)...")

# Use CountVectorizer (better for LDA than TF-IDF)
count_vectorizer = CountVectorizer(stop_words="english", max_features=3000)
X_counts = count_vectorizer.fit_transform(df["clean_text"].sample(10000, random_state=42))  # sample for speed

# Train LDA with 20 topics
lda = LatentDirichletAllocation(n_components=20, random_state=42, learning_method="batch")
lda.fit(X_counts)

# Extract top keywords for each topic
terms = count_vectorizer.get_feature_names_out()
topic_keywords = {}

def get_topic_keywords(model, feature_names, n_top_words=10):
    keywords = []
    for topic_idx, topic in enumerate(model.components_):
        top_features = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        keywords.append(top_features)
    return keywords

keywords_per_topic = get_topic_keywords(lda, terms, n_top_words=10)

print("\n📌 LDA Topics (Top 10 Keywords Each):")
for i, words in enumerate(keywords_per_topic):
    print(f"Topic {i}: {', '.join(words)}")
    topic_keywords[i] = ", ".join(words)

# ------------------------------------------------------------
# 4. User Input Prediction
# ------------------------------------------------------------
print("\n📝 Try it yourself! (Example inputs below)")
print("👉 Example 1: 'The government passed a new law regarding healthcare reform.'")
print("👉 Example 2: 'The football team won their final match of the season.'")
print("👉 Example 3: 'Scientists discovered a new exoplanet with potential life.'")

user_text = input("\nUser enters a news article: ")

if user_text.strip() != "":
    # Clean user input
    user_clean = clean_text(user_text)

    # ---- Supervised prediction ----
    user_tfidf = tfidf.transform([user_clean])
    predicted_category = clf.predict(user_tfidf)[0]

    # ---- Unsupervised topic prediction ----
    user_counts = count_vectorizer.transform([user_clean])
    user_topic_dist = lda.transform(user_counts)[0]
    predicted_topic = user_topic_dist.argmax()

    # ---- Display results ----
    print("\n📌 Predicted Results:")
    print(f"🟢 Supervised Category (Trained Labels): {predicted_category}")
    print(f"🔵 Unsupervised Topic (LDA): Topic {predicted_topic}")
    print(f"   🔑 Topic Keywords: {topic_keywords[predicted_topic]}")

else:
    print("⚠️ No input provided. Please enter some news text.")
