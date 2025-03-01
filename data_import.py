import csv

f = open('datasets/chess_games.csv', 'r')
matches = csv.reader(f)

for i in range(100):
    print(f[i])