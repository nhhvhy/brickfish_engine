import chess
import chess.pgn
import sqlite3
import io
import numpy as np
import re
import pandas as pd
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Global Variables
letter_2_num = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7}
num_2_letter = {0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h'}

# Board to tensor
def board_2_rep(board):
    pieces = ['p','r','n','b','q','k']
    layers = [create_rep_layer(board, piece) for piece in pieces]
    return np.stack(layers)

def create_rep_layer(board, piece):
    b = str(board)
    b = re.sub(f'[^ {piece}{piece.upper()}\n]', '.', b)
    b = re.sub(f'{piece}', '-1', b)
    b = re.sub(f'{piece.upper()}', '1', b)
    b = re.sub(r'\.', '0', b)
    
    mat = [list(map(int, row.split())) for row in b.split('\n')]
    return np.array(mat)

# Convert PGN to UCI
def move_2_rep(move, board):
    try:
        board.push_san(move)
        move = str(board.pop())

        from_output_layer = np.zeros((8,8))
        to_output_layer = np.zeros((8,8))

        from_row, from_col = 8 - int(move[1]), letter_2_num[move[0]]
        to_row, to_col = 8 - int(move[3]), letter_2_num[move[2]]

        from_output_layer[from_row, from_col] = 1
        to_output_layer[to_row, to_col] = 1

        return np.stack([from_output_layer, to_output_layer])
    
    except chess.InvalidMoveError:
        print(f"Warning: Invalid move encountered -> {move}. Skipping...")
        return None  # Skip this move


def create_move_list(b):
    return re.sub(r'\d*\. ', '', b).split(' ')[:-1]

# Load dataset
chess_data_raw = pd.read_csv('datasets/chess_games.csv', usecols=['AN', 'WhiteElo'])
chess_data = chess_data_raw[chess_data_raw['WhiteElo'] > 2000]
del chess_data_raw
gc.collect()

# PyTorch Dataset
class ChessDataset(Dataset):
    def __init__(self, games):
        self.games = games.reset_index(drop=True)  # Reset index to avoid errors

    def __len__(self):
        return len(self.games)

    def __getitem__(self, index):
        try:
            random_game = self.games.iloc[index]  # Directly use the given index
            moves = create_move_list(random_game)
            if len(moves) < 2:  # Ensure enough moves exist
                return self.__getitem__(np.random.randint(len(self.games)))

            game_state_i = np.random.randint(len(moves) - 1)
            next_move = moves[game_state_i]
            moves_before = moves[:game_state_i]

            board = chess.Board()
            for move in moves_before:
                try:
                    board.push_san(move)
                except chess.InvalidMoveError:
                    return self.__getitem__(np.random.randint(len(self.games)))  # Skip bad samples

            x = board_2_rep(board)
            y = move_2_rep(next_move, board)

            if y is None:  # Skip invalid moves
                return self.__getitem__(np.random.randint(len(self.games)))

            if game_state_i % 2 == 1:
                x *= -1

            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

        except Exception as e:
            print(f"Dataset error at index {index}: {e}")
            return self.__getitem__(np.random.randint(len(self.games)))  # Fallback to a valid sample


# DataLoader
data_train = ChessDataset(chess_data['AN'])
data_train_loader = DataLoader(data_train, batch_size=32, shuffle=True, drop_last=True)

# Convolutional Residual Module
class Module(nn.Module):
    def __init__(self, hidden_size):
        super(Module, self).__init__()
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.bn2 = nn.BatchNorm2d(hidden_size)
        self.activation = nn.SELU()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return self.activation(x + residual)

# ChessNet Model
class ChessNet(nn.Module):
    def __init__(self, hidden_layers=4, hidden_size=200):
        super(ChessNet, self).__init__()
        self.hidden_layers = hidden_layers
        self.input_layer = nn.Conv2d(6, hidden_size, 3, stride=1, padding=1)
        self.residual_blocks = nn.ModuleList([Module(hidden_size) for _ in range(hidden_layers)])
        self.output_layer = nn.Conv2d(hidden_size, 2, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        for layer in self.residual_blocks:
            x = layer(x)
        return self.output_layer(x)

# Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChessNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training Loop
num_epochs = 5
for epoch in range(num_epochs):
    total_loss = 0
    for x_batch, y_batch in data_train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        output = model(x_batch)

        # Loss Calculation
        loss_from = criterion(output[:, 0, :, :].view(-1, 8*8), y_batch[:, 0, :, :].view(-1, 8*8))
        loss_to = criterion(output[:, 1, :, :].view(-1, 8*8), y_batch[:, 1, :, :].view(-1, 8*8))
        loss = loss_from + loss_to

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), "models/chessnet_model_2.pth")
print("Model saved successfully!")
