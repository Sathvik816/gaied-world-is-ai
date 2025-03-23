# Step 1: Install Required Libraries
#for colab !pip install transformers torch datasets pandas
#for local pip install transformers torch datasets pandas

# Step 2: Import Necessary Libraries
import os
import torch
import email
import pandas as pd
import numpy as np
from email import policy
from email.parser import BytesParser
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from google.colab import files

os.environ["WANDB_DISABLED"] = "true"

# Step 3: Create EML File Directory
eml_folder_path = "/content/eml_files"
os.makedirs(eml_folder_path, exist_ok=True)

print(f"✅ Directory created: {eml_folder_path}")

# Step 4: Upload & Parse Emails
uploaded_files = files.upload()  # Upload EML files
for file_name in uploaded_files.keys():
    os.rename(file_name, os.path.join(eml_folder_path, file_name))

print("✅ EML files uploaded successfully!")

# Get list of EML files
eml_files = [os.path.join(eml_folder_path, f) for f in os.listdir(eml_folder_path) if f.endswith(".eml")]


def extract_email_content(eml_file):
    with open(eml_file, 'rb') as f:
        msg = email.message_from_binary_file(f, policy=policy.default)
        # Ensure we extract text properly (handles list payloads)
        if msg.is_multipart():
            return ''.join(part.get_payload(decode=True).decode(errors='ignore') for part in msg.walk() if
                           part.get_content_type() == "text/plain")
        else:
            return msg.get_payload(decode=True).decode(errors='ignore')


email_texts = [extract_email_content(file) for file in eml_files if os.path.exists(file)]

# Remove empty emails
email_texts = [text.strip() for text in email_texts if text.strip()]
print(f"✅ Extracted {len(email_texts)} emails successfully!")

# Step 5: Define Labels & Create DataFrame
categories = ["General Inquiry", "Payment Issue", "Fraud Alert", "Adjustment", "Money Movement-Inbound"]

# Ensure correct label generation
num_samples = len(email_texts)
labels = np.random.randint(0, len(categories), num_samples)  # Random labels for demo

df = pd.DataFrame({"text": email_texts, "label": labels})
print("✅ DataFrame created!")

# Step 6: Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Split into Train & Test sets
dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Step 7: Tokenize Data
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(categories))


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# Tokenize dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Convert labels to integer format
tokenized_datasets = tokenized_datasets.map(lambda x: {"label": int(x["label"])})

# Step 8: Define Training Arguments & Trainer
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"]
)

# Train the model
print("🚀 Training model...")
trainer.train()


# Step 9: Email Classification Function
def classify_email(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length")

    with torch.no_grad():  # Avoids unnecessary gradient calculations
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits).item()
    return categories[prediction]


# Example classification
new_email = f"""Description: Facility lender share adjustment

Borrower: XYZ Limited LLC
Deal Name: XYZ Limited LLC $40 16-4-2023

Effective 11-Jan-2024, the lender shares of facility Loan have been adjusted.
Your share of the commitment was USD 597. It has increased to USD 637



Regards,  
Kane Nick
Telephone #:
"""
print(f"🔹 Classified as: {classify_email(new_email)}")

