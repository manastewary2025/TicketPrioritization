import joblib
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load models and mappings - LDA START
try:
    vectorizer = joblib.load("vectorizer.pkl")
    lda = joblib.load("lda_model.pkl")
    priority_mapping = joblib.load("priority_mapping.pkl")
    print("Models and mappings loaded successfully.")
except Exception as e:
    raise ValueError(f"Error loading models or mappings: {e}")
# Load models and mappings - LDA END

# Load models and mappings - Logistic Regression START
# try:
#     vectorizer = joblib.load("vectorizer.pkl")
#     logreg = joblib.load("logreg_model.pkl")  
#     priority_mapping = joblib.load("priority_mapping.pkl")
#     print("Models and mappings loaded successfully.")
# except Exception as e:
#     raise ValueError(f"Error loading models or mappings: {e}")
# Load models and mappings - Logistic Regression END

# Reverse mapping for priority labels
reverse_priority_mapping = {v: k for k, v in priority_mapping.items()}

# Topic Prediction Function
def predict_topic(text):
    try:
        text_vector = vectorizer.transform([text])
        #LDA START
        topic_distribution = lda.transform(text_vector)
        return topic_distribution.argmax()
        #LDA END

        #Logistic Regression START
        # Logistic Regression direct prediction returns array of class labels
        # predicted_label = logreg.predict(text_vector)[0]
        # return predicted_label
        #Logistic Regression END
    except Exception as e:
        raise ValueError(f"Error predicting topic: {e}")

# Prioritize Ticket Function
def prioritize_ticket(text):
    predicted_topic = predict_topic(text)
    return reverse_priority_mapping.get(predicted_topic, "Unknown Priority")

# Evaluate Predictions and Generate Confusion Matrix
def evaluate_predictions(test_texts, test_labels):
    try:
        predicted_labels = [predict_topic(text) for text in test_texts]

        # Ensure the labels align with reverse_priority_mapping
        labels = sorted(reverse_priority_mapping.keys())  # Sorted numeric labels
        target_names = [reverse_priority_mapping[label] for label in labels]

        # Classification report
        # print("\nClassification Report:")
        # print(classification_report(test_labels, predicted_labels, labels=labels, target_names=target_names))

        # Confusion matrix
        # conf_matrix = confusion_matrix(test_labels, predicted_labels, labels=labels)
        # print("\nConfusion Matrix:")
        # print(conf_matrix)

        # print("\nClassification Report:")
        # print(classification_report(test_labels, predicted_labels, target_names=reverse_priority_mapping.values()))
        
        # conf_matrix = confusion_matrix(test_labels, predicted_labels)
        # print("\nConfusion Matrix:")
        # print(conf_matrix)

        # Plot confusion matrix
        # plt.figure(figsize=(8, 6))
        # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
        #             xticklabels=target_names, 
        #             yticklabels=target_names)
        # plt.title("Confusion Matrix")
        # plt.xlabel("Predicted")
        # plt.ylabel("Actual")
        # plt.show()
    except Exception as e:
        raise ValueError(f"Error during evaluation: {e}")

# Test example
test_text = "TIFFANY |TCGDBPMAP001|WINDOWS|MSSQL:Job-Exec-Time : Value|master|1"
try:
    print(f"Predicted Priority: {prioritize_ticket(test_text)}")
except Exception as e:
    print(f"Error: {e}")

# Example test dataset
test_texts = [
    "email service not working",
    "unable to login to the portal",
    "application crashing on checkout",
    "slow response from the server",
    "data not syncing with database"
]
test_labels = [2, 3, 2, 4, 3]  # Replace with actual numeric labels

# Evaluate predictions
try:
    evaluate_predictions(test_texts, test_labels)
except Exception as e:
    print(f"Error: {e}")


