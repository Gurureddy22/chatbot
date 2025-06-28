from datasets import load_dataset
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import pickle

# Load dataset from Hugging Face
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

# Convert dataset to DataFrame
df = pd.DataFrame(dataset["train"])

# Debug: Print dataset columns
print("Dataset Columns:", df.columns)
print(df.head())  # Print the first few rows for inspection

# Try auto-detecting the correct column names
expected_cols = ["question", "intent", "response"]
if not all(col in df.columns for col in expected_cols):
    print("Dataset does not have expected column names. Attempting renaming...")
    df.rename(columns={df.columns[1]: "question", df.columns[3]: "intent", df.columns[4]: "response"}, inplace=True)

# Check again if necessary columns exist
if "question" not in df.columns or "intent" not in df.columns or "response" not in df.columns:
    raise ValueError("Dataset must contain 'question', 'intent', and 'response' columns.")

# Extract user queries and intents
X = df["question"]
y = df["intent"]

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer()
X_vectors = vectorizer.fit_transform(X)

# Train SVM model for intent classification
clf = SVC(kernel="linear")
clf.fit(X_vectors, y)

# Save trained model and vectorizer
with open("chatbot_model.pkl", "wb") as model_file:
    pickle.dump(clf, model_file)

with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

# Save responses dictionary
responses = df.groupby("intent")["response"].apply(list).to_dict()
with open("responses.pkl", "wb") as resp_file:
    pickle.dump(responses, resp_file)

print("✅ Chatbot model trained successfully!")
