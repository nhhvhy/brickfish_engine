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

# DB - Maybe try and implement later?
'''
# DB Setup
db = sqlite3.connect('matches.db')
cursor = db.cursor()
statement = 'SELECT AN FROM Matches WHERE WhiteElo BETWEEN 0 AND 1200 LIMIT 1' # Remove 'LIMIT _' during actual training runs
cursor.execute(statement)
output = cursor.fetchall()


# Match Import Setup
matches = []
for pgn in output:
    pgn = str(pgn[0])
    pgn = io.StringIO(pgn)
    match = chess.pgn.read_game(pgn)
    matches.append(match)
'''

# Global Vars
letter_2_num = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7}
num_2_letter = {0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h'}

''' Testing
match = matches[0]
board = match.board()
print(board)

for move in match.mainline_moves():
    board.push(move)
    print(board)

'''

# Board to tensor
def board_2_rep(board):
    pieces = ['p','r','n','b','q','k']
    layers = []
    for piece in pieces:
        layers.append(create_rep_layer(board, piece))
    board_rep = np.stack(layers)
    
    return board_rep

def create_rep_layer(board, type):
    b = str(board)
    b = re.sub(f'[^{type}{type.upper()} \n]', '.', b)
    b = re.sub(f'{type}', '-1', b)
    b = re.sub(f'{type.upper()}', '1', b)
    b = re.sub(r'\.', '0', b)

    mat = []
    for row in b.split('\n'):
        row = row.split(' ')
        row = [int(x) for x in row]
        mat.append(row)

    return np.array(mat)

# Convert PGN to UCI
def move_2_rep(move, board):

    board.push_san(move).uci()
    move = str(board.pop())

    from_output_layer = np.zeros((8,8))
    from_row = 8 - int(move[1])
    from_column = letter_2_num[move[0]]
    from_output_layer[from_row, from_column] = 1

    to_output_layer = np.zeros((8,8))
    to_row = 8 - int(move[3])
    to_column = letter_2_num[move[2]]
    to_output_layer[to_row, to_column] = 1

    return np.stack([from_output_layer, to_output_layer])


def create_move_list(b):
    return re.sub('\d*\. ','',b).split(' ')[:-1]



# Data Import
chess_data_raw = pd.read_csv('datasets/chess_games.csv', usecols=['AN', 'WhiteElo'])
chess_data = chess_data_raw[chess_data_raw['WhiteElo'] < 1000]
del chess_data_raw
gc.collect() # All your RAM are belong to us

## PYTORCH ##
class ChessDataset(Dataset):
    
    def __init__(self, games):
        super(ChessDataset, self).__init__()
        self.games = games

    def __len__(self):
        return 69420

    def __getitem__(self, index):
        game_i = np.random.randint(self.games.shape[0])
        random_game = chess_data['AN'].values[game_i]
        moves = create_move_list(random_game)
        game_state_i = np.random.randint(len(moves) - 1)
        next_move = moves[game_state_i]
        moves = moves[:game_state_i]
        board = chess.Board()
        for move in moves:
            board.push_san(move)
        x = board_2_rep(board)
        y = move_2_rep(next_move, board)
        if game_state_i % 2 == 1:
            x *= -1
        return x, y


data_train = ChessDataset(chess_data['AN'])
data_train_loader = DataLoader(data_train, batch_size=32, shuffle=True, drop_last=True)


## CNN ##

class module(nn.Module):

    def __init__(self, hidden_size):
        super(module, self).__init__()
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.bn2 = nn.BatchNorm2d(hidden_size)
        self.activation1 = nn.SELU()
        self.activation2 = nn.SELU()

    def forward(self, x):
        x_input = torch.clone(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + x_input
        x = self.activation2(x)
        return x

class ChessNet(nn.Module):

    def __init__(self, hidden_layers=4, hidden_size=200):
        super(PolicyNet, self).__init__()
        self.hidden_layers = hidden_layers
        self.input_layer = nn.Conv2d(6, hidden_size, 3, stride=1, padding=1)
        self.module_list = nn.ModuleList([module(hidden_size) for i in range(hidden_layers)])
        self.output_layer = nn.Conv2d(hidden_size, 2, 3, stride=1, padding=1)

    def forward(self, x):

        x = self.input_layer(x)
        x = F.relu(x)

        for i in range(self.hidden_layers):
            x = self.module_list[i](x)

        x = self.output_layer(x)

        return x
    
''' Not Sure Where This Goes

metric_from = nn.CrossEntropyLoss()
metric_to = nn.CrossEntropyLoss()

loss_from = metric_from(output[:,0,:], y[:,0,:])
loss_to = metric_to(output[:,1,:], y[:,1,:])
loss = loss_from + loss_to

'''

'''
db.commit()
db.close()
'''