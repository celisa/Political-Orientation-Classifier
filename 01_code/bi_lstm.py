import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import TensorDataset, DataLoader

#remember to cite:
#pytorch docs
#QUESTIONS
#1. what is padding_idx? When we do padding, we add zeros to the beginning of the sentences to make them all the same length. Why not at the end?
#2 What does dropout do? What do the values mean?
#2. what is proj_size? 
#3. is the one-hot encoding correct? Should we do embedding with GloVe? But that would generate 2d arrays for each sentence, right? Seems like a lot to process.
#4 what does .contiguous().view(-1, self.hidden_dim) do?
"""We need to add an embedding layer because there are less words in our vocabulary. 
It is massively inefficient to one-hot encode that many classes. So, instead of one-hot encoding, 
we can have an embedding layer and use that layer as a lookup table. 
You could train an embedding layer using Word2Vec, then load it here. 
But, it's fine to just make a new layer, using it for only dimensionality reduction, and let the network learn the weights."""


class BiLSTM(nn.Module):

    #initialization
    def __init__(self, vocab_size ,embedding_dim = 64, hidden_size = 128, num_layers = 1, bias = True, batch_first = True, dropout = 0.1, bidirectional = True, proj_size = 0, num_classes = 2, batch_size = 100, output_dim = 1):

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

        #inherit from nn.Module
        super(BiLSTM, self).__init__()

        #create embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim) #what is padding_idx? vocab_size is the size of one-hot dict + 1 for padding.

        #Define the LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, bias, batch_first, dropout, bidirectional, proj_size)

        # Define the output layer
        self.out_linear = nn.Linear(hidden_size * 2, hidden_size) #originally was input_size instead of hidden_size
        self.linear = nn.Linear(hidden_size * 2, output_dim) #should this be multiplied by 2?

    def init_hidden(self): 
        """Initialize the hidden state of the LSTM"""
        # returns (a, b), where a and b both have shape (num_layers, batch_size, hidden_size) of zeros.
        return (torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_size),
                torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_size))
    
    def forward(self, input):

        #get batch size
        batch_size = input.size(0)

        #initialize hidden state
        h, c = self.init_hidden()

        # Embed the input
        embedded = self.embedding(input) #embedded is a 3d array with shape (batch_size, seq_len, embedding_dim)
        
        # Forward pass through LSTM layer
        output, (h_, c_) = self.lstm(embedded, (h, c))
        output = output.contiguous().view(-1, self.hidden_size * 2) #what does this do??

        output = nn.Dropout(self.dropout)(output)
        output = self.linear(output)

        # sigmoid function
        output = nn.Sigmoid()(output)

        # reshape to be batch_size
        output = output.view(batch_size, -1)
        y_pred = output[:, -1] # get last batch of labels
        return y_pred

def tokenize(data):
    # Tokenize data
    pass

def add_padding(sentences, seq_len = 280):
    """adds zeros to the beginning of the sentences to make them all the same length"""
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

def get_input_data(path):
    """Reads in the data and returns the input data and labels for training and testing"""
    # Read train data

    # create Tensor datasets
    X_train = None
    y_train = None
    X_test = None
    y_test = None

    #create Tensor datasets
    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_data, shuffle=True, batch_size=100)
    valid_loader = DataLoader(test_data, shuffle=True, batch_size=100)
    pass

def calculate_f1_score(y_pred, y_true):
    """Calculates the f1 score"""
    pass

def calculate_accuracy(y_pred, y_true):
    """Calculates the accuracy"""
    pass

def train_model(model, train_loader, valid_loader, epochs = 10):
    clip = 5
epochs = 5 
valid_loss_min = np.Inf
# train for some number of epochs
epoch_tr_loss,epoch_vl_loss = [],[]
epoch_tr_acc,epoch_vl_acc = [],[]

for epoch in range(epochs):
    train_losses = []
    train_acc = 0.0
    model.train()
    # initialize hidden state 
    h = model.init_hidden(batch_size)
    for inputs, labels in train_loader:
        
        inputs, labels = inputs.to(device), labels.to(device)   
        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])
        
        model.zero_grad()
        output,h = model(inputs,h)
        
        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        train_losses.append(loss.item())
        # calculating accuracy
        accuracy = acc(output,labels)
        train_acc += accuracy
        #`clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
 
    
        
    val_h = model.init_hidden(batch_size)
    val_losses = []
    val_acc = 0.0
    model.eval()
    for inputs, labels in valid_loader:
            val_h = tuple([each.data for each in val_h])

            inputs, labels = inputs.to(device), labels.to(device)

            output, val_h = model(inputs, val_h)
            val_loss = criterion(output.squeeze(), labels.float())

            val_losses.append(val_loss.item())
            
            accuracy = acc(output,labels)
            val_acc += accuracy
            
    epoch_train_loss = np.mean(train_losses)
    epoch_val_loss = np.mean(val_losses)
    epoch_train_acc = train_acc/len(train_loader.dataset)
    epoch_val_acc = val_acc/len(valid_loader.dataset)
    epoch_tr_loss.append(epoch_train_loss)
    epoch_vl_loss.append(epoch_val_loss)
    epoch_tr_acc.append(epoch_train_acc)
    epoch_vl_acc.append(epoch_val_acc)
    print(f'Epoch {epoch+1}') 
    print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
    print(f'train_accuracy : {epoch_train_acc*100} val_accuracy : {epoch_val_acc*100}')
    if epoch_val_loss <= valid_loss_min:
        torch.save(model.state_dict(), '../working/state_dict.pt')
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,epoch_val_loss))
        valid_loss_min = epoch_val_loss
    print(25*'==')
    

def run_model():
    # Define hyperparameters
    vocab_size = None #size of one-hot dict + 1 for padding
    
    # Instantiate the model w/ hyperparams
    model = BiLSTM(vocab_size)
    print(model)

    # Define Loss, Optimizer
    # loss and optimization functions
    lr=0.001

    criterion = nn.BCELoss() #binary cross entropy loss

    optimizer = torch.optim.Adam(model.parameters(), lr=lr) #Adam optimizer

    # Train the model
    train_model(model, train_loader, valid_loader, epochs = 10)

    return model


def predict_text(text, model):
    """Predicts the sentiment of the text using the model"""
        word_seq = np.array([vocab[preprocess_string(word)] for word in text.split() 
                         if preprocess_string(word) in vocab.keys()])
        word_seq = np.expand_dims(word_seq,axis=0)
        pad =  torch.from_numpy(padding_(word_seq,500))
        inputs = pad.to(device)
        batch_size = 1
        h = model.init_hidden(batch_size)
        h = tuple([each.data for each in h])
        output, h = model(inputs, h)
        return(output.item())

