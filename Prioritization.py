import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE  # For balancing dataset --Synthetic Minority Over-sampling Technique
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import joblib


# Load datasets
data1 = pd.read_excel("Incident_2.xlsx")
data2 = pd.read_excel("Incidents.xlsx")
data = pd.concat([data1, data2], ignore_index=True).drop_duplicates()

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')  # For additional WordNet languages

# Ensure Priority column exists
if 'Priority' not in data.columns:
    raise ValueError("The dataset must contain a 'Priority' column.")

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Combine relevant text columns for analysis
data['clean_text'] = (
    data['Description'].fillna('') + ' ' +
    data['Short Description'].fillna('') + ' ' +
    data['Comments and Work notes'].fillna('')
)

# Text preprocessing: Lowercasing, stopword removal, lemmatization
def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Normalization: Remove special characters and numbers
    text = ''.join([char if char.isalpha() or char.isspace() else ' ' for char in text])
    # Tokenization
    words = text.split()
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    # Lemmatization
    words = [lemmatizer.lemmatize(word) for word in words]
    # Stemming
    words = [stemmer.stem(word) for word in words]

    return ' '.join(words)

data['clean_text'] = data['clean_text'].str.replace(r'[^\w\s]', '', regex=True).apply(preprocess_text)

# Relative pruning: Remove words that appear too infrequently (e.g., frequency < 2)
def prune_text(text, min_word_freq=2):
    word_freq = pd.Series(text.split()).value_counts()
    pruned_words = [word for word in text.split() if word_freq[word] >= min_word_freq]
    return ' '.join(pruned_words)

data['clean_text'] = data['clean_text'].apply(lambda x: prune_text(x, min_word_freq=2))

# Map priorities to numeric labels
priority_mapping = {priority: idx for idx, priority in enumerate(data['Priority'].unique())}
data['priority_label'] = data['Priority'].map(priority_mapping)

# # Train-test split
# train_texts, test_texts, train_labels, test_labels = train_test_split(
#     data['clean_text'], data['priority_label'], test_size=0.2, random_state=42
# )

# Train-test split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data['clean_text'], data['priority_label'], test_size=0.2, random_state=42 
    ,stratify=data['priority_label']
)

# Apply SMOTE to balance the classes in the training data
# smote = SMOTE(random_state=42)
# train_texts_resampled, train_labels_resampled = smote.fit_resample(
#     train_texts.values.reshape(-1, 1), train_labels
# )

# # Vectorization
# vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
# train_vectors = vectorizer.fit_transform(train_texts)
# test_vectors = vectorizer.transform(test_texts)

# Vectorization
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', max_features=5000)
train_vectors = vectorizer.fit_transform(train_texts)
test_vectors = vectorizer.transform(test_texts)

# Resample the training labels using SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
train_texts_resampled, train_labels_resampled = smote.fit_resample(train_vectors, train_labels)


# Verify the resampling result
print(f"Original training labels distribution: {train_labels.value_counts()}")
print(f"Resampled training labels distribution: {pd.Series(train_labels_resampled).value_counts()}")


# # LDA Model --Start
lda = LatentDirichletAllocation(n_components=len(priority_mapping), random_state=42)
# Train the model
print("Starting model training with LDA...")
#lda.fit(train_vectors)
lda.fit(train_texts_resampled)
print("Model training completed.")

# Save the models and mapping
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(lda, "lda_model.pkl") 
joblib.dump(priority_mapping, "priority_mapping.pkl")
# # LDA Model --END

# A supervised classifier LogisticRegression --START
# logreg = LogisticRegression(
#     random_state=42,
#     max_iter=1000  # Increase if needed for convergence
# )
# print("Starting model training with Logistic Regression...")
# logreg.fit(train_texts_resampled, train_labels_resampled)
# print("Model training completed.")

# joblib.dump(vectorizer, "vectorizer.pkl")
# joblib.dump(logreg, "logreg_model.pkl")  # <--- Save Logistic Regression
# joblib.dump(priority_mapping, "priority_mapping.pkl")
# A supervised classifier LogisticRegression --END

print("Models and mapping saved.")

# Predict and Evaluate
# def predict_priorities(model, vectors):
#     """Get predicted priorities from the LDA model."""
#     topic_distributions = model.transform(vectors)
#     predicted_labels = topic_distributions.argmax(axis=1)
#     return predicted_labels

# # Predictions
# train_predictions = predict_priorities(lda, train_vectors)
# test_predictions = predict_priorities(lda, test_vectors)

# # Evaluation on Training Data
# print("\nClassification Report (Training Data):")
# print(classification_report(train_labels, train_predictions))

# # Confusion Matrix for Training Data
# train_conf_matrix = confusion_matrix(train_labels, train_predictions)
# print("\nTraining Data Confusion Matrix:")
# print(train_conf_matrix)

# # Plot Confusion Matrix for Training Data
# plt.figure(figsize=(8, 6))
# sns.heatmap(train_conf_matrix, annot=True, fmt='d', cmap='Greens', xticklabels=priority_mapping.keys(), yticklabels=priority_mapping.keys())
# plt.title("Training Data Confusion Matrix")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.show()


# # Evaluation on Test Data
# print("\nClassification Report (Test Data):")
# print(classification_report(test_labels, test_predictions))

# # Confusion Matrix for Test Data
# test_conf_matrix = confusion_matrix(test_labels, test_predictions)
# print("\nTest Data Confusion Matrix:")
# print(test_conf_matrix)

# # Plot Confusion Matrix for Test Data
# plt.figure(figsize=(8, 6))
# sns.heatmap(test_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=priority_mapping.keys(), yticklabels=priority_mapping.keys())
# plt.title("Test Data Confusion Matrix")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.show()