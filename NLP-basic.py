import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data import Field, LabelField, BucketIterator

# Define the fields
TEXT = Field(tokenize="spacy")
LABEL = LabelField(dtype=torch.float)

# Load the IMDB dataset
train_data, test_data = IMDB.splits(TEXT, LABEL)

# Build the vocabulary
TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d")
LABEL.build_vocab(train_data)

# Define the model
class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.rnn(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        output = self.fc(hidden)
        return output

# Define the hyperparameters
vocab_size = len(TEXT.vocab)
embedding_dim = 100
hidden_dim = 256
output_dim = 1
batch_size = 64

# Define the model, optimizer, and loss function
model = SentimentClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

# Define the iterators
train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=batch_size,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    sort_key=lambda x: len(x.text),
    sort_within_batch=True,
)

# Train the model
model.train()
for epoch in range(10):
    for batch in train_iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        loss.backward()
        optimizer.step()

# Evaluate the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for batch in test_iterator:
        predictions = model(batch.text).squeeze(1)
        rounded_predictions = torch.round(torch.sigmoid(predictions))
        correct += (rounded_predictions == batch.label).sum().item()
        total += len(batch.label)
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")
```

This code defines a simple N
