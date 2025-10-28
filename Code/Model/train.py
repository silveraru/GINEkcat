import json
import math
import numpy as np
import pickle
import torch
from sklearn.metrics import mean_squared_error, r2_score
from torch import nn, optim
import timeit
from Code.Model.GINEkcat import GINEkcat

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, num_epochs=200, early_stopping_patience=10):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            for data in self.train_loader:
                inputs, targets = data
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_train_loss = total_loss / len(self.train_loader)
            val_loss = self.validate()
            print(f'Epoch {epoch+1}, Training Loss: {avg_train_loss}, Validation Loss: {val_loss}')
            self.check_early_stopping(val_loss)

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data in self.val_loader:
                inputs, targets = data
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def check_early_stopping(self, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.early_stopping_counter = 0
            # Save the model here
        else:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= self.early_stopping_patience:
                print('Early stopping triggered')
                return

class Validator:
    def __init__(self, model, val_loader):
        self.model = model
        self.val_loader = val_loader

    def validate(self):
        self.model.eval()
        predictions, targets = [], []
        with torch.no_grad():
            for data in self.val_loader:
                inputs, target = data
                outputs = self.model(inputs)
                predictions.append(outputs)
                targets.append(target)
        return predictions, targets

class Tester:
    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader

    def test(self):
        self.model.eval()
        predictions, targets = [], []
        with torch.no_grad():
            for data in self.test_loader:
                inputs, target = data
                outputs = self.model(inputs)
                predictions.append(outputs)
                targets.append(target)
        return predictions, targets

if __name__ == '__main__':
    # Load DeepEnzyme dictionaries
    with open('path/to/fingerprint_dict.pkl', 'rb') as f:
        fingerprint_dict = pickle.load(f)
    with open('path/to/word_dict.pkl', 'rb') as f:
        word_dict = pickle.load(f)

    # Load DeepEnzyme tensors
    fingerprints = torch.tensor(np.load('path/to/fingerprints.npy'))
    smileadjacencies = torch.tensor(np.load('path/to/smileadjacencies.npy'))
    sequences = torch.tensor(np.load('path/to/sequences.npy'))
    proteinadjacencies = torch.tensor(np.load('path/to/proteinadjacencies.npy'))
    logkcat = torch.tensor(np.load('path/to/logkcat.npy'))

    n_fingerprint = fingerprint_dict['n_fingerprint']
    model = GINEkcat(n_fingerprint=n_fingerprint, word_dict=word_dict)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Prepare data loaders
    # train_loader, val_loader, test_loader = ... (your data loading logic here)

    trainer = Trainer(model, train_loader, val_loader, optimizer, criterion)
    trainer.train()