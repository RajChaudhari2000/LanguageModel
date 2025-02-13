import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModel
import torch.nn as nn
import torch.optim as optim
import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit
from SPARQLWrapper import SPARQLWrapper, JSON
import spacy

# Load dataset from DBpedia
dataset = load_dataset("dbpedia_14")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_data(data):
    texts = [entry["content"] for entry in data]
    tokenized_texts = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    return tokenized_texts["input_ids"], tokenized_texts["attention_mask"]

# Process Train Data
train_data, train_masks = preprocess_data(dataset["train"])

# Define Custom Transformer-based Language Model
class CustomTransformerModel(nn.Module):
    def __init__(self, model_name, output_dim):
        super(CustomTransformerModel, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.fc = nn.Linear(self.transformer.config.hidden_size, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        output = self.fc(outputs.last_hidden_state[:, -1, :])
        return self.softmax(output)

# Initialize Model
model_name = "bert-base-uncased"
vocab_size = tokenizer.vocab_size
model = CustomTransformerModel(model_name, vocab_size)

# Training Setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, data, masks, epochs=3):
    model.train()
    for epoch in range(epochs):
        for batch, mask in zip(data, masks):
            optimizer.zero_grad()
            outputs = model(batch, mask)
            loss = criterion(outputs, batch[:, 1:])
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Train the Model
train_model(model, train_data, train_masks)

# Save the Model
torch.save(model.state_dict(), "dbpedia_transformer_model.pth")

# Load NLP Model
nlp = spacy.load("en_core_web_sm")

def query_dbpedia(entity):
    sparql = SPARQLWrapper("https://dbpedia.org/sparql")
    sparql.setQuery(f"""
    SELECT ?subject ?predicate ?object WHERE {{
      ?subject ?predicate ?object .
      FILTER (regex(str(?subject), "{entity}", "i"))
    }} LIMIT 10
    """)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    triplets = [(res["subject"]["value"], res["predicate"]["value"], res["object"]["value"]) for res in results["results"]["bindings"]]
    return triplets

def triplet_to_natural(triplets):
    return " ".join([f"{s.replace('_', ' ')} {p.replace('_', ' ')} {o.replace('_', ' ')}." for s, p, o in triplets])

class ChatBot(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("DBpedia Chatbot")
        self.setGeometry(200, 200, 600, 400)
        layout = QVBoxLayout()
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.input = QLineEdit()
        self.input.setPlaceholderText("Ask me anything...")
        self.input.returnPressed.connect(self.handle_query)
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.handle_query)
        layout.addWidget(QLabel("DBpedia Chatbot"))
        layout.addWidget(self.chat_history)
        layout.addWidget(self.input)
        layout.addWidget(self.send_button)
        self.setLayout(layout)

    def handle_query(self):
        question = self.input.text().strip()
        if not question:
            return
        self.chat_history.append(f"You: {question}")
        doc = nlp(question)
        entity = [ent.text for ent in doc.ents]
        triplets = query_dbpedia(entity[0]) if entity else []
        response = triplet_to_natural(triplets) if triplets else "I couldn't find an answer."
        self.chat_history.append(f"Bot: {response}\n")
        self.input.clear()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    chatbot = ChatBot()
    chatbot.show()
    sys.exit(app.exec())
