import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import F1Score
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

#remember to cite:
#pytorch docs
#QUESTIONS
#1. what is padding_idx? When we do padding, we add zeros to the beginning of the sentences to make them all the same length. Why not at the end?
#2 What does dropout do? What do the values mean?
#2. what is proj_size? 
#3. is the one-hot encoding correct? Should we do embedding with GloVe? But that would generate 2d arrays for each sentence, right? Seems like a lot to process.
#4 what does .contiguous().view(-1, self.hidden_dim) do? 
#4 how do we choose the embedding size for the model?
#5. what does clip do for the optimizer? Helps with explording gradient problem
#6 what happens when model(output, hidden) is called? how is tied to forward?
#7 how to calculate f1 scores? What is the model actually outputting?? I need to understand the output of the model better.
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
    
    def forward(self, input, hidden):

        #get batch size
        batch_size = input.size(0)

        # Embed the input
        embedded = self.embedding(input) #embedded is a 3d array with shape (batch_size, seq_len, embedding_dim)
        
        # Forward pass through LSTM layer
        output, hidden = self.lstm(embedded, hidden)
        output = output.contiguous().view(-1, self.hidden_size * 2) #what does this do??

        output = nn.Dropout(self.dropout)(output)
        output = self.linear(output)

        # sigmoid function
        output = nn.Sigmoid()(output)

        # reshape to be batch_size
        output = output.view(batch_size, -1)
        y_pred = output[:, -1] # get last batch of labels
        return y_pred, hidden

def create_one_hot_dict(df):
    """Creates a dictionary that maps words to integers"""
    #flatten the list of lists
    corpus = df['text'].tolist()
    corpus = [item for sublist in corpus for item in sublist]
    corpus = Counter(corpus) #in case I want to grab the most common words

    #create a dictionary that maps words to integers
    one_hot_dict = {word: i + 1 for i, word in enumerate(corpus)}
    return one_hot_dict

def tokenize(df, one_hot_dict):
    """Tokenizes the text in the dataframe"""

    #tokenize the text
    tokenized_data = []
    for tweet in df['text']:
        tokenized_data.append([one_hot_dict[word] for word in tweet]) #have to additional checks if only taking n most popular words
    
    return np.array(tokenized_data)

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
    train_df = pd.read_csv(path + 'cleaned_train.csv')
    X_train = pd.DataFrame(train_df['text'])
    y_train = train_df['labels'].to_numpy()
    print(f"length of training data: {len(y_train)}")

    # Read test data
    test_df = pd.read_csv(path + 'cleaned_test.csv')
    X_test = pd.DataFrame(test_df['text'])
    y_test = test_df['labels'].to_numpy()
    print(f"length of testing data: {len(y_test)}")
    
    # Tokenize data
    one_hot_dict_train = create_one_hot_dict(X_train)
    one_hot_dict_test = create_one_hot_dict(X_test)
    X_train = tokenize(X_train, one_hot_dict_train)
    X_test = tokenize(X_test, one_hot_dict_test)
    X_train = add_padding(X_train)
    X_test = add_padding(X_test)

    # create Tensor datasets
    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_data, shuffle=True, batch_size=100)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=100)
    return train_loader, test_loader, len(one_hot_dict_train) + 1

def calculate_f1_score(y_pred, y_true): #DEBUG
    """Calculates the f1 score"""
    
    #calculate f1 score
    f1 = F1Score(task="binary")
    return f1(y_true, y_pred).item() #not sure if I should call .item()

def calculate_accuracy(y_pred, y_true):
    """Calculates the accuracy"""
    predictions = torch.round(y_pred.squeeze())  # rounds to the nearest integer
    return torch.sum(predictions == y_true).item()

def train_model(model, device, train_loader, test_loader, clip = 5, epochs = 10, lr = 0.05):
    """Trains the model and returns the training and testing losses and accuracies"""
    epoch_train_losses = []
    epoch_test_losses = []
    epoch_train_acc = []
    epoch_test_acc = []
    epoch_train_f1 = []
    epoch_test_f1 = []

    #binary cross entropy loss
    loss_func = nn.BCELoss()

    #Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # initialize model_loss_criteria
    min_loss = np.Inf

    for epoch in range(epochs):
        train_losses = []
        test_losses = []
        train_acc = 0.0
        test_acc = 0.0
        train_f1 = 0.0
        test_f1 = 0.0

        #training
        model.train()
        hidden = model.init_hidden()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            hidden = tuple([each.data for each in hidden])

            model.zero_grad()
            output,hidden = model(inputs, hidden)

            #calculate loss
            loss = loss_func(output.squeeze(), labels.float())
            loss.backward()
            train_losses.append(loss.item())

            #calculate accuracy
            accuracy = calculate_accuracy(output, labels)
            train_acc += accuracy

            #calculate f1 score
            #f1 = calculate_f1_score(output, labels)
            #train_f1 += f1

            #clip the gradient to prevent exploding gradient
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        
        #testing
        hidden_test = model.init_hidden()
        model.eval()
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            hidden_test = tuple([each.data for each in hidden_test])

            output, hidden_test = model(inputs, hidden_test)

            #calculate loss
            loss = loss_func(output.squeeze(), labels.float())
            test_losses.append(loss.item())

            #calculate accuracy
            accuracy = calculate_accuracy(output, labels)
            test_acc += accuracy

            #calculate f1 score
            #f1 = calculate_f1_score(output, labels)
            #test_f1 += f1
        
        #calculate average loss, accuracy, and f1 score
        epoch_train_losses.append(np.mean(train_losses))
        epoch_test_losses.append(np.mean(test_losses))
        epoch_train_acc.append(train_acc / len(train_loader.dataset))
        epoch_test_acc.append(test_acc / len(test_loader.dataset))
        #epoch_train_f1.append(train_f1/len(train_loader.dataset))
        #epoch_test_f1.append(test_f1/len(test_loader.dataset))

        print(f'Epoch: {epoch+1}/{epochs}')
        print(f'Train Loss: {epoch_train_losses[-1]:.3f} | Train Acc: {epoch_train_acc[-1]:.3f} | Train F1: TBD')
        print(f'Test Loss: {epoch_test_losses[-1]:.3f} | Test Acc: {epoch_test_acc[-1]:.3f} | Test F1: TBD')

        #save model
        if epoch_test_losses[-1] <= min_loss:
            torch.save(model.state_dict(), './saved_models/model.pt')
            print("model saved")
            min_loss = epoch_test_losses[-1]
        print(30*'=')

    return epoch_train_losses, epoch_test_losses, epoch_train_acc, epoch_test_acc, epoch_train_f1, epoch_test_f1
             
def run_model():

    # Get the data
    train_loader, test_loader, vocab_size = get_input_data('/workspaces/NLP_FinalProject/00_source_data/')
    
    # Instantiate the model w/ hyperparams
    model = BiLSTM(vocab_size)
    print(model)

    #check if cuda is available
    cuda_avail = torch.cuda.is_available()
    if cuda_avail:
        device = torch.device("cuda")
        print("using GPU for training")
    else:
        device = torch.device("cpu")
        print("using CPU for training")
    model.to(device)

    # Train the model
    epoch_train_losses, epoch_test_losses, epoch_train_acc, epoch_test_acc, epoch_train_f1, epoch_test_f1 = train_model(model, device, train_loader, test_loader, clip = 5, epochs = 10, lr = 0.05)

    return epoch_train_losses, epoch_test_losses, epoch_train_acc, epoch_test_acc, epoch_train_f1, epoch_test_f1

def predict_text(text, model): #TOCOMPLETE - SECONDARY
    """Predicts the sentiment of the text using the model"""
    pass

def plot_results(epoch_train_losses, epoch_test_losses, epoch_train_acc, epoch_test_acc, epoch_train_f1, epoch_test_f1): #TOCOMPLETE - SECONDARY
    """Plots the training and testing losses and accuracies"""
    pass

def main():
    """Main function"""
    epoch_train_losses, epoch_test_losses, epoch_train_acc, epoch_test_acc, epoch_train_f1, epoch_test_f1 = run_model()
    pass
    
if __name__ == '__main__':
    main()