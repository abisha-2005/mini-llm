import nltk
import torch
import torch.nn as nn

from nltk.tokenize import word_tokenize

nltk.download('punkt')

# -----------------------------------
# LOAD DATA AGAIN
# -----------------------------------

with open("data.txt","r") as file:

    text=file.read().lower()

tokens=word_tokenize(text)

vocab=sorted(list(set(tokens)))

word_to_idx={

    word:i

    for i,word in enumerate(vocab)

}

idx_to_word={

    i:word

    for word,i in word_to_idx.items()

}

# -----------------------------------
# MODEL
# -----------------------------------

class MiniLLM(nn.Module):

    def __init__(

        self,

        vocab_size,

        embedding_dim

    ):

        super().__init__()

        self.embedding=nn.Embedding(

            vocab_size,

            embedding_dim

        )

        self.fc1=nn.Linear(

            embedding_dim*2,

            128

        )

        self.relu=nn.ReLU()

        self.fc2=nn.Linear(

            128,

            vocab_size

        )

    def forward(

        self,

        x

    ):

        x=self.embedding(x)

        x=x.view(

            x.size(0),

            -1

        )

        x=self.fc1(x)

        x=self.relu(x)

        x=self.fc2(x)

        return x

# -----------------------------------
# LOAD MODEL
# -----------------------------------

model=MiniLLM(

    len(vocab),

    20

)

model.load_state_dict(

    torch.load(

        "mini_llm.pth"

    )

)

model.eval()

# -----------------------------------
# PREDICT
# -----------------------------------

def predict_next_word(

    word1,

    word2

):

    x=torch.tensor([

        [

            word_to_idx[word1],

            word_to_idx[word2]

        ]

    ])

    with torch.no_grad():

        output=model(x)

        prediction=torch.argmax(

            output

        ).item()

    return idx_to_word[prediction]

# -----------------------------------
# TEST
# -----------------------------------

while True:

    first=input(

        "\nFirst Word : "

    ).lower()

    second=input(

        "Second Word : "

    ).lower()

    print(

        "Prediction :",

        predict_next_word(

            first,

            second

        )

    )