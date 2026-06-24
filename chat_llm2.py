import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F

from nltk.tokenize import word_tokenize

# -----------------------------------
# DOWNLOAD TOKENIZER
# -----------------------------------

nltk.download('punkt')
nltk.download('punkt_tab')

# -----------------------------------
# LOAD DATA (must match train_llm.py exactly!)
# -----------------------------------

with open("data.txt", "r") as file:
    lines = file.readlines()

text = " eos ".join(line.strip().lower() for line in lines if line.strip())
text += " eos"

tokens = word_tokenize(text)

# -----------------------------------
# VOCABULARY
# -----------------------------------

vocab = sorted(list(set(tokens)))

word_to_idx = {
    word: i
    for i, word in enumerate(vocab)
}

idx_to_word = {
    i: word
    for word, i in word_to_idx.items()
}

# -----------------------------------
# MODEL DEFINITION
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
# LOAD SAVED MODEL
# -----------------------------------

model = MiniLLM(len(vocab), 20)
model.load_state_dict(torch.load("mini_llm.pth"))
model.eval()

print("=" * 50)
print("MODEL LOADED SUCCESSFULLY")
print("=" * 50)

# -----------------------------------
# PREDICT FUNCTION (with temperature sampling)
# -----------------------------------

def predict_next_word(word1, word2, temperature=0.8):

    if word1 not in word_to_idx:
        return None
    if word2 not in word_to_idx:
        return None

    x = torch.tensor([[word_to_idx[word1], word_to_idx[word2]]])

    with torch.no_grad():
        output = model(x)
        probs = F.softmax(output / temperature, dim=-1)
        prediction = torch.multinomial(probs, num_samples=1).item()

    return idx_to_word[prediction]

# -----------------------------------
# GENERATE FULL SENTENCE (stops at "eos")
# -----------------------------------

def generate(word1, word2, length=15, temperature=0.8):

    result = [word1, word2]

    for _ in range(length):

        next_word = predict_next_word(result[-2], result[-1], temperature)

        if next_word is None or next_word == "eos":
            break

        result.append(next_word)

    return " ".join(result)

# -----------------------------------
# TEST PREDICTIONS
# -----------------------------------

print("\n")
print("=" * 50)
print("MODEL TESTING")
print("=" * 50)

print("artificial intelligence ->", generate("artificial", "intelligence"))
print("machine learning ->", generate("machine", "learning"))
print("deep learning ->", generate("deep", "learning"))

# -----------------------------------
# USER INTERACTION
# -----------------------------------

while True:

    print("\n")

    first = input("First Word (or exit): ").lower()

    if first == "exit":
        print("Goodbye!")
        break

    second = input("Second Word: ").lower()

    if first not in word_to_idx or second not in word_to_idx:
        print("\nOne or both words not in vocabulary, try again.")
        continue

    sentence = generate(first, second, length=15)

    print("\nGenerated Sentence:", sentence)