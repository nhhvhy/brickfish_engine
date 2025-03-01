import chess
import chess.pgn
import sqlite3
import io
import numpy as np
import re

# DB Setup
db = sqlite3.connect('matches.db')
cursor = db.cursor()
statement = 'SELECT AN FROM Matches WHERE WhiteElo BETWEEN 0 AND 1200 LIMIT 1' # Remove 'LIMIT _' during actual training runs
cursor.execute(statement)
output = cursor.fetchall()

# Global Vars
letter_2_num = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7}
num_2_letter = {0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h'}


# Match Import Setup
matches = []
for pgn in output:
    pgn = str(pgn[0])
    pgn = io.StringIO(pgn)
    match = chess.pgn.read_game(pgn)
    matches.append(match)


''' Testing
match = matches[0]
board = match.board()
print(board)

for move in match.mainline_moves():
    board.push(move)
    print(board)

'''

# Board to 3D array
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
    b = re.sub(f'\.', '0', b)

    mat = []
    for row in b.split('\n'):
        row = row.split(' ')
        row = [int(x) for x in row]
        mat.append(row)

    return np.array(mat)

db.commit()
db.close()
