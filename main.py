
from backEnd import *


def main():
    game = BrickFish()
    print(game.legal_moves())
    print(game.board_state())

if __name__ == "__main__":
    main()