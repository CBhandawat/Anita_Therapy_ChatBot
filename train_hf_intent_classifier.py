import json
import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# 1. Load and preprocess
with open("data/intents.json") as f:
    intents = json.load(f)["intents"]

texts, labels, label2id = [], [], {}
for i, intent in enumerate(intents):
    label2id[intent["tag"]] = i
    for pattern in intent["patterns"]:
        texts.append(pattern)
        labels.append(i)

id2label = {v: k for k, v in label2id.items()}

# 2. Prepare DataFrame and split
df = pd.DataFrame({"text": texts, "label": labels})
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

# 3. Convert to Hugging Face Datasets
train_ds = Dataset.from_pandas(train_df)
eval_ds = Dataset.from_pandas(eval_df)

# 4. Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

train_ds = train_ds.map(tokenize, batched=True)
eval_ds = eval_ds.map(tokenize, batched=True)

train_ds = train_ds.rename_column("label", "labels")
eval_ds = eval_ds.rename_column("label", "labels")

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
eval_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 5. Load model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# 6. Define metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
    }

# 7. Training arguments
training_args = TrainingArguments(
    output_dir="./model/hf_intent",
    eval_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    logging_steps=10,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True
)

# 8. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    compute_metrics=compute_metrics
)

# 9. Train and Evaluate
trainer.train()
metrics = trainer.evaluate()
print("Evaluation Results:", metrics)

# 10. Save model and tokenizer
model.save_pretrained("model/hf_intent")
tokenizer.save_pretrained("model/hf_intent")

# 11. Save label maps
with open("model/intent_id_map.json", "w") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f)
