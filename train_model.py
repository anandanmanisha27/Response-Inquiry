import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel
import torch
from torch.nn.functional import normalize

# Sample data (40 examples: ~13 per class)
data = [
    ("I have a problem with your app crashing", "Technical Support"),
    ("It’s not working after the update", "Technical Support"),
    ("The app shows error code 502", "Technical Support"),
    ("I can't log into my account", "Technical Support"),
    ("Screen goes blank when I open settings", "Technical Support"),
    ("My password reset link doesn’t work", "Technical Support"),
    ("App keeps logging me out", "Technical Support"),
    ("Notifications stopped working suddenly", "Technical Support"),
    ("It’s showing a blank white screen", "Technical Support"),
    ("I'm getting an error while uploading a file", "Technical Support"),
    ("Installation fails on Windows 11", "Technical Support"),
    ("I need help migrating from another tool", "Technical Support"),
    ("The search feature isn't returning results", "Technical Support"),

    ("Can you add dark mode feature?", "Product Feature Request"),
    ("Please include integration with Slack", "Product Feature Request"),
    ("Can you add more analytics to the dashboard?", "Product Feature Request"),
    ("I’d love to have a widget for home screen", "Product Feature Request"),
    ("Could you add Google Calendar sync?", "Product Feature Request"),
    ("Please add a Linux version", "Product Feature Request"),
    ("Is it possible to export data to Excel?", "Product Feature Request"),
    ("Voice command support would be awesome", "Product Feature Request"),
    ("Support for multiple profiles, please", "Product Feature Request"),
    ("Add biometric login support", "Product Feature Request"),
    ("Can you integrate with Zapier?", "Product Feature Request"),
    ("Offline mode would be useful", "Product Feature Request"),
    ("I’d like reminders and task scheduling", "Product Feature Request"),

    ("What’s your pricing model?", "Sales Lead"),
    ("Do you offer enterprise plans?", "Sales Lead"),
    ("I'd like to know more about your enterprise package", "Sales Lead"),
    ("How can I contact a sales representative?", "Sales Lead"),
    ("Do you provide discounts for startups?", "Sales Lead"),
    ("Do you offer monthly billing?", "Sales Lead"),
    ("How much does it cost for a team of 10?", "Sales Lead"),
    ("Can we get a demo before purchasing?", "Sales Lead"),
    ("Is there an API available?", "Sales Lead"),
    ("Can I upgrade to the pro version later?", "Sales Lead"),
    ("Do you support white-label options?", "Sales Lead"),
    ("We’re considering your product for our business", "Sales Lead"),
    ("Can I talk to someone from your sales team?", "Sales Lead"),
]

# Split into inputs and labels
X_texts, y_labels = zip(*data)

# Label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_labels)

# Load BERT model and tokenizer
print("🔄 Loading BERT...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# Convert texts to BERT embeddings (CLS token)
def get_bert_embeddings(texts):
    with torch.no_grad():
        inputs = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=128)
        outputs = bert_model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        return normalize(cls_embeddings).numpy()

print("🔄 Generating embeddings...")
X_bert = get_bert_embeddings(X_texts)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_bert, y, test_size=0.2, random_state=42, stratify=y)

# Train classifier
print("🚀 Training classifier...")
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate
print("\n📊 Evaluation:")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save model and label encoder
joblib.dump(clf, "bert_model.joblib")
joblib.dump(label_encoder, "label_encoder.joblib")
print("\n✅ Model and label encoder saved (bert_model.joblib, label_encoder.joblib).")
