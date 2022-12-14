"""
Description: This script trains a bidirectional LSTM model to predict the political affiliation of the user based on their tweet.
Libraries used:
    - PyTorch
    @incollection{NEURIPS2019_9015,
    title = {PyTorch: An Imperative Style, High-Performance Deep Learning Library},
    author = {Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and Desmaison, Alban and Kopf, Andreas and Yang, Edward and DeVito, Zachary and Raison, Martin and Tejani, Alykhan and Chilamkurthy, Sasank and Steiner, Benoit and Fang, Lu and Bai, Junjie and Chintala, Soumith},
    booktitle = {Advances in Neural Information Processing Systems 32},
    pages = {8024--8035},
    year = {2019},
    publisher = {Curran Associates, Inc.},
    url = {http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf}
    }
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
import warnings
from sklearn.metrics import accuracy_score, f1_score
from torch.cuda.amp import GradScaler, autocast
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

warnings.filterwarnings("ignore")

#declare global variable
batch_size_global = 10

class BiLSTM(nn.Module):

    # initialization
    def __init__(
        self,
        vocab_size,
        embedding_dim=300,
        hidden_size=128,
        num_layers=1,
        bias=True,
        batch_first=True,
        dropout=0.25,
        bidirectional=True,
        proj_size=0,
        num_classes=2,
        batch_size=batch_size_global,
        output_dim=1
    ):

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.proj_size = proj_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.output_dim = output_dim

        # inherit from nn.Module
        super(BiLSTM, self).__init__()

        # create embedding layer
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim
        )  # vocab_size is the size of one-hot dict + 1 for padding.

        # Define the LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
            proj_size,
        )

        # Define the output layer
        self.linear = nn.Linear(
            hidden_size * 2, output_dim
        )  

    def init_hidden(self, device):
        """Initialize the hidden state of the LSTM"""
        # returns (a, b), where a and b both have shape (num_layers, batch_size, hidden_size) of zeros.
        return (
            torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_size, device=device), #should these num layers be multiplied by 2? trying without
            torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_size, device=device),
        )

    def forward(self, input, hidden):
        """Forward pass through the network. Called with self.forward"""

        # get batch size
        batch_size = input.size(0)

        # Embed the input
        embedded = self.embedding(
            input
        )  # embedded is a 3d array with shape (batch_size, seq_len, embedding_dim)

        # Forward pass through LSTM layer (propagate input through the LSTM layer)
        output, hidden = self.lstm(embedded, hidden)
        output = output.contiguous().view(
            -1, self.hidden_size * 2
        )  #reshape the data for Dense layer. Is this necessary?

        # Use dropout before the final layer to improve with regularization
        output = nn.Dropout(self.dropout)(output)
        output = self.linear(output)

        # sigmoid function
        output = nn.Sigmoid()(output)

        # reshape to be batch_size
        output = output.view(batch_size, -1)
        y_pred = output[:, -1]  # get last batch of labels
        return y_pred, hidden

def get_one_hot_encodings(df, col_name):
    """One hot encode a column of a dataframe into embedding matrix"""
    
    matrix = np.array(df[col_name]).tolist()

    flat = [x for row in matrix for x in row]
    unique_words = set(flat)
    vocab = {x: i for i, x in enumerate(unique_words)}
    idxss = [[vocab[word] for word in words] for words in matrix]
    embedding = np.zeros((len(idxss), len(vocab)), dtype=np.uint8)

    for i, idxs in enumerate(idxss):
        embedding[i, idxs] = 1

    return embedding, unique_words

def get_input_data(path):
    """Reads in the data and returns the input data and labels for training and testing"""
    # Read train data
    train_df = pd.read_csv(path + "cleaned_train.csv")
    # convert the 'text_new' field to a list of words - currently in str
    train_df['text_new'] = train_df['text_new'].apply(lambda x: x.strip('][').replace("'",'').split(', '))
    train_df['text'] = train_df['text_new'].apply(lambda x: ' '.join(x))
    #X_train = pd.DataFrame(train_df["text_new"])
    y_train = train_df["labels"].to_numpy()
    print(f"length of training data: {len(y_train)}")

    # Read test data
    test_df = pd.read_csv(path + "cleaned_test.csv")
    test_df['text_new'] = test_df['text_new'].apply(lambda x: x.strip('][').replace("'",'').split(', '))
    test_df['text'] = test_df['text_new'].apply(lambda x: ' '.join(x))
    #X_test = pd.DataFrame(test_df["text_new"])
    y_test = test_df["labels"].to_numpy()
    print(f"length of testing data: {len(y_test)}")

    # Tokenize data
    #one_hot_dict_train = create_one_hot_dict(X_train)
    #one_hot_dict_test = create_one_hot_dict(X_test)
    #X_train = tokenize(X_train, one_hot_dict_train)
    #X_test = tokenize(X_test, one_hot_dict_test)
    #X_train = add_padding(X_train)
    #X_test = add_padding(X_test)
    #X_train, train_vocab = get_one_hot_encodings(X_train, "text_new")
    #X_test, test_vocab = get_one_hot_encodings(X_test, "text_new")

    # Initizalize the vectorizer with max nr words and ngrams (1: single words, 2: two words in a row)
    vectorizer_tfidf_train = TfidfVectorizer(max_features=15000, ngram_range=(1,2))
    vectorizer_tfidf_test = TfidfVectorizer(max_features=15000, ngram_range=(1,2))
    
    # Fit the vectorizer to the training data
    X_train = train_df['text'].to_list()
    X_train = vectorizer_tfidf_train.fit_transform(X_train)
    X_train = X_train.toarray()

    train_vocab = set(vectorizer_tfidf_train.get_feature_names_out())

    # Fit the vectorizer to the testing data
    X_test = test_df['text'].to_list()
    X_test = vectorizer_tfidf_test.fit_transform(X_test)
    X_test = X_test.toarray()

    test_vocab = set(vectorizer_tfidf_test.get_feature_names_out())

    # create Tensor datasets
    #train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_data = TensorDataset(torch.tensor(X_train).to(torch.int64), torch.from_numpy(y_train))
    #test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_data = TensorDataset(torch.tensor(X_test).to(torch.int64), torch.from_numpy(y_test))

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size_global, drop_last=True)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size_global, drop_last=True)
    return train_loader, test_loader, len(train_vocab.union(test_vocab)) + 1


def calculate_f1_score(y_pred, y_true, device):
    """Calculates the f1 score"""
    f1 = f1_score(y_true.cpu().data, y_pred.cpu().data > 0.5, average='binary')
    return f1


def calculate_accuracy(y_pred, y_true):
    """Calculates the accuracy"""
    predictions = torch.round(y_pred.squeeze())  # rounds to the nearest integer
    return torch.sum(predictions == y_true).item()


def train_model(model, device, train_loader, test_loader, clip=5, epochs=10, lr=0.05):
    """Trains the model and returns the training and testing losses and accuracies"""
    epoch_train_losses = []
    epoch_test_losses = []
    epoch_train_acc = []
    epoch_test_acc = []
    epoch_train_f1 = []
    epoch_test_f1 = []

    # binary cross entropy loss
    loss_func = nn.BCELoss()

    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # initialize model_loss_criteria
    min_loss = np.Inf

    for epoch in range(epochs):
        train_losses = []
        test_losses = []
        train_acc = 0.0
        test_acc = 0.0
        train_f1 = []
        test_f1 = []

        # training
        model.train()
        hidden = model.init_hidden(device)
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if (inputs.shape[0], inputs.shape[1]) != (
                model.batch_size,
                inputs.shape[1],
            ):
                print("Validation - Input Shape Issue:", inputs.shape)
                continue
            hidden = tuple([each.data for each in hidden])

            model.zero_grad()
            output, hidden = model(inputs, hidden)

            # calculate loss
            loss = loss_func(output.squeeze(), labels.float())
            loss.backward()
            train_losses.append(loss.item())

            # calculate accuracy
            accuracy = calculate_accuracy(output, labels)
            train_acc += accuracy

            # calculate f1 score
            f1 = calculate_f1_score(output, labels, device)
            train_f1.append(f1)

            # clip the gradient to prevent exploding gradient
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        # testing
        hidden_test = model.init_hidden(device)
        model.eval()
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if (inputs.shape[0], inputs.shape[1]) != (
                model.batch_size,
                inputs.shape[1],
            ):
                print("Validation - Input Shape Issue:", inputs.shape)
                continue
            hidden_test = tuple([each.data for each in hidden_test])

            output, hidden_test = model(inputs, hidden_test)

            # calculate loss
            loss = loss_func(output.squeeze(), labels.float())
            test_losses.append(loss.item())

            # calculate accuracy
            accuracy = calculate_accuracy(output, labels)
            test_acc += accuracy

            # calculate f1 score
            f1 = calculate_f1_score(output, labels, device)
            test_f1.append(f1)

        # calculate average loss, accuracy, and f1 score
        epoch_train_losses.append(np.mean(train_losses))
        epoch_test_losses.append(np.mean(test_losses))
        epoch_train_acc.append(train_acc / len(train_loader.dataset))
        epoch_test_acc.append(test_acc / len(test_loader.dataset))
        epoch_train_f1.append(np.mean(train_f1))
        epoch_test_f1.append(np.mean(test_f1))

        print(f"Epoch: {epoch+1}/{epochs}")
        print(
            f"Train Loss: {epoch_train_losses[-1]:.3f} | Train Acc: {epoch_train_acc[-1]:.3f} | Train F1: {epoch_train_f1[-1]:.3f}"
        )
        print(
            f"Test Loss: {epoch_test_losses[-1]:.3f} | Test Acc: {epoch_test_acc[-1]:.3f} | Test F1: {epoch_test_f1[-1]:.3f}"
        )

        # save model
        if epoch_test_losses[-1] <= min_loss:
            torch.save(model.state_dict(), "./saved_models/model.pt")
            print("model saved")
            min_loss = epoch_test_losses[-1]
        print(50 * "=")

    return (
        epoch_train_losses,
        epoch_test_losses,
        epoch_train_acc,
        epoch_test_acc,
        epoch_train_f1,
        epoch_test_f1,
    )


def run_model():

    # Get the data
    train_loader, test_loader, vocab_size = get_input_data(
        "/workspaces/nlp-training-pt2/00_source_data/" # REMEMBER TO CHANGE THIS WHEN YOU RUN THE CODE IN THE CORRECT REPO
    )

    # Instantiate the model w/ hyperparams
    model = BiLSTM(vocab_size)
    print(model)
    print(50 * "=")

    # check if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Device: {device}")
    print(50 * "=")

    # Train the model
    (
        epoch_train_losses,
        epoch_test_losses,
        epoch_train_acc,
        epoch_test_acc,
        epoch_train_f1,
        epoch_test_f1,
    ) = train_model(
        model, device, train_loader, test_loader, clip=5, epochs=10, lr=0.05
    )

    return (
        epoch_train_losses,
        epoch_test_losses,
        epoch_train_acc,
        epoch_test_acc,
        epoch_train_f1,
        epoch_test_f1,
    )


def predict_text(text, model):  # TOCOMPLETE - SECONDARY
    """Predicts the sentiment of the text using the model"""
    pass


def plot_results(
    epoch_train_losses,
    epoch_test_losses,
    epoch_train_acc,
    epoch_test_acc,
    epoch_train_f1,
    epoch_test_f1,
):  # TOCOMPLETE - SECONDARY
    """Plots the training and testing losses and accuracies"""
    pass


def main():
    """Main function"""
    epoch_train_losses, epoch_test_losses, epoch_train_acc, epoch_test_acc, epoch_train_f1, epoch_test_f1 = run_model()
    
    pd.DataFrame(epoch_train_losses).to_csv("./epoch_train_losses.csv", index=False)
    pd.DataFrame(epoch_test_losses).to_csv("./epoch_test_losses.csv", index=False)
    pd.DataFrame(epoch_train_acc).to_csv("./epoch_train_acc.csv", index=False)
    pd.DataFrame(epoch_test_acc).to_csv("./epoch_test_acc.csv", index=False)
    pd.DataFrame(epoch_train_f1).to_csv("./epoch_train_f1.csv", index=False)
    pd.DataFrame(epoch_test_f1).to_csv("./epoch_test_f1.csv", index=False)
    pass


if __name__ == "__main__":
    main()

def tokenize(df, one_hot_dict): #retired
    """Tokenizes the text in the dataframe"""

    # tokenize the text
    tokenized_data = []
    for tweet in df["text"]:
        tokenized_data.append(
            [one_hot_dict[word] for word in tweet]
        )  # have to additional checks if only taking n most popular words

    return np.array(tokenized_data)


def add_padding(sentences, seq_len=50): #retired
    """adds zeros to the beginning of the sentences to make them all the same length"""
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for ii, tweet in enumerate(sentences):
        if len(tweet) != 0 and len(tweet) <= seq_len:
            features[ii, -len(tweet) :] = np.array(tweet)[:seq_len]
        else:  # if the tweet is longer than seq_len, then just take the last seq_len words
            features[ii, -seq_len:] = np.array(tweet)[:seq_len]
    return features

def create_one_hot_dict(df): #retired
    """Creates a dictionary that maps words to integers"""
    # flatten the list of lists
    corpus = df["text"].tolist()
    corpus = [item for sublist in corpus for item in sublist]
    corpus = Counter(corpus)  # in case I want to grab the most common words

    # create a dictionary that maps words to integers
    one_hot_dict = {word: i + 1 for i, word in enumerate(corpus)}
    return one_hot_dict
