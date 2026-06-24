import nltk
import torch
import torch.nn as nn
import torch.optim as optim

from nltk.tokenize import word_tokenize

# -----------------------------------
# DOWNLOAD TOKENIZER
# -----------------------------------

nltk.download('punkt')
nltk.download('punkt_tab')

# -----------------------------------
# LOAD DATA
# -----------------------------------

with open("data.txt", "r") as file:
    lines = file.readlines()

# Join lines with a plain "eos" marker (no symbols, so tokenizer won't split it)
text = " eos ".join(line.strip().lower() for line in lines if line.strip())
text += " eos"

print("="*50)
print("DATASET")
print("="*50)

print(text)

# -----------------------------------
# TOKENIZATION
# -----------------------------------

tokens = word_tokenize(text)

print("\n")
print("="*50)
print("TOKENS")
print("="*50)

print(tokens)

# -----------------------------------
# VOCABULARY
# -----------------------------------

vocab = sorted(list(set(tokens)))

print("\n")
print("="*50)
print("VOCABULARY")
print("="*50)

print(vocab)

# -----------------------------------
# WORD TO INDEX
# -----------------------------------

word_to_idx = {
    word: i
    for i, word in enumerate(vocab)
}

idx_to_word = {
    i: word
    for word, i in word_to_idx.items()
}

print("\n")
print("="*50)
print("WORD TO INDEX")
print("="*50)

print(word_to_idx)

# -----------------------------------
# CREATE TRAINING DATA
# -----------------------------------

X = []
y = []

for i in range(len(tokens) - 2):

    input_words = [
        word_to_idx[tokens[i]],
        word_to_idx[tokens[i+1]]
    ]

    target_word = word_to_idx[tokens[i+2]]

    X.append(input_words)
    y.append(target_word)

X = torch.tensor(X)
y = torch.tensor(y)

print("\n")
print("="*50)
print("INPUT SHAPE")
print("="*50)

print(X.shape)

print("\n")
print("="*50)
print("TARGET SHAPE")
print("="*50)

print(y.shape)

# -----------------------------------
# MODEL
# -----------------------------------

class MiniLLM(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# -----------------------------------
# CREATE MODEL
# -----------------------------------

vocab_size = len(vocab)

model = MiniLLM(vocab_size, embedding_dim=20)

print("\n")
print("="*50)
print("MODEL CREATED")
print("="*50)

print(model)

# -----------------------------------
# LOSS
# -----------------------------------

loss_function = nn.CrossEntropyLoss()

# -----------------------------------
# OPTIMIZER
# -----------------------------------

optimizer = optim.Adam(model.parameters(), lr=0.05)

# -----------------------------------
# TRAINING
# -----------------------------------

print("\n")
print("="*50)
print("TRAINING STARTED")
print("="*50)

epochs = 1000

for epoch in range(epochs):

    outputs = model(X)
    loss = loss_function(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss = {loss.item():.4f}")

print("\n")
print("="*50)
print("TRAINING COMPLETED")
print("="*50)

# -----------------------------------
# SAVE MODEL
# -----------------------------------

torch.save(model.state_dict(), "mini_llm.pth")

print("Model Saved")