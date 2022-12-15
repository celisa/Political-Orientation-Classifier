import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from bi_lstm import BiLSTM


# Read in the data

def load_data():
    """Helper function to load the metrics data from the results folder"""

    test_acc = pd.read_csv('./results/real_data/epoch_test_acc.csv')
    train_acc = pd.read_csv('./results/real_data/epoch_train_acc.csv')
    test_f1 = pd.read_csv('./results/real_data/epoch_test_f1.csv')
    train_f1 = pd.read_csv('./results/real_data/epoch_train_f1.csv')
    train_loss = pd.read_csv('./results/real_data/epoch_train_losses.csv')
    test_loss = pd.read_csv('./results/real_data/epoch_test_losses.csv')
    return test_acc, train_acc, test_f1, train_f1, train_loss, test_loss

def generate_plot(data, title, metric):
    """Helper function to generate the plots for the metrics"""
    test_data = data[1]
    train_data = data[0]
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(x=train_data.index, y="0",data=train_data, label = "training")
    ax2 = sns.lineplot(x=test_data.index, y="0",data=test_data, label = "test")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    plt.legend(loc="upper left")
    plt.savefig(f'./plots/{title}.png')

def prepare_data():
    #prepare the data
    real_data = pd.read_csv('../00_source_data/cleaned_test.csv').sample(100)
    real_data['text_new'] = real_data['text_new'].apply(lambda x: x.strip('][').replace("'",'').split(', '))
    real_data['text'] = real_data['text_new'].apply(lambda x: ' '.join(x))
    y_real = real_data["labels"].to_numpy()
    vectorizer_tfidf_real = TfidfVectorizer(max_features=40000, ngram_range=(1,2))
    X_real = vectorizer_tfidf_real.fit_transform(real_data['text'].to_list())
    X_real = X_real.toarray()

    synth_data = pd.read_csv('../00_source_data/synthetic_data.csv').sample(100)
    y_synth = synth_data["labels"].to_numpy()
    vectorizer_tfidf_synth = TfidfVectorizer(max_features=40000, ngram_range=(1,2))
    X_synth = vectorizer_tfidf_synth.fit_transform(synth_data['Tweets'].to_list())
    X_synth = X_synth.toarray()
    
    #return real_data, synth_data
    return torch.tensor(X_real).to(torch.int64), y_real,torch.tensor(X_synth).to(torch.int64), y_synth

def generate_confusion_matrix(X, y, title, is_synth = False):
    """Helper function to generate the confusion matrix"""

    device = torch.device('cpu' if torch.cuda.is_available() else 'gpu')

    #load the model
    model = BiLSTM(18565) #vocab size
    model.load_state_dict(torch.load('./saved_models/model_real.pt'))

    model = model.to(device) # Set model to gpu
    model.eval()

    inputs = X
    labels = y

    inputs = inputs.to(device) # You can move your input to gpu, torch defaults to cpu
    #initialize hidden state
    hidden = (
            torch.zeros(2, len(inputs), 128, device=device), 
            torch.zeros(2, len(inputs), 128, device=device),
        )

    # Run forward pass
    with torch.no_grad():
        hidden = tuple([each.data for each in hidden])
        pred, hidden = model(inputs, hidden)

    # Do something with pred
    pred = torch.round(pred.squeeze())  # rounds to the nearest integer
    pred = pred.cpu().data  # remove from computational graph to cpu and as numpy

     #confusion matrix
    cm = confusion_matrix(labels, pred)
    print(cm)
    
    sns.heatmap(cm.T, annot=True, fmt='d', cbar=False)

    plt.title(title)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    if is_synth:
        plt.savefig("./plots/confusion_matrix_synth.png")
    else:
        plt.savefig("./plots/confusion_matrix_real.png")
    plt.clf()


def main():
    test_acc, train_acc, test_f1, train_f1, train_loss, test_loss = load_data()
    generate_plot([train_acc, test_acc], 'Accuracy for Training and Test Data - Real', 'Accuracy')
    generate_plot([train_f1, test_f1], 'F1 Score for Training and Test Data - Real', 'F1 Score')
    generate_plot([train_loss, test_loss], 'Loss for Training and Test Data - Real', 'Loss')
    X_real, y_real, X_synth, y_synth  = prepare_data()
    generate_confusion_matrix(X_real, y_real, "Confusion Matrix for LSTM - Real Data")
    generate_confusion_matrix(X_synth, y_synth, "Confusion Matrix for LSTM - Synthetic Data", is_synth = True)

if __name__ == '__main__':
    main()
