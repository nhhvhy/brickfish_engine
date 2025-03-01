import chess
import json

class BrickFish:
    def __init__(self):
        self.board = chess.Board()
        # print(self.board)  # Display initial board

    def move_piece(self, move_uci):
        """Executes a move if legal."""
        move = chess.Move.from_uci(move_uci)
        if move in self.board.legal_moves:
            self.board.push(move)
            return True
        else:
            print("Illegal move")
            return False

    def legal_moves(self, pos_start=None):
        if not pos_start:
            moves = [move.uci() for move in self.board.legal_moves]
        else:
            pos_square = chess.parse_square(pos_start)
            moves = [move.uci() for move in self.board.legal_moves if move.from_square == pos_square]

        return json.dumps({"legal_moves": moves}, indent=4)
    
    def board_state(self):
        return json.dumps({"board state" : repr(self.board).split("'").pop(1).split(' ').pop(0)}, indent=4)
    
def main():
    game = BrickFish()
    
    print(game.board_state())
    print(game.legal_moves("e2"))
    game.move_piece("e2e4")
    print(game.board_state())
    print(game.legal_moves("e2"))

if __name__ == "__main__":
    main()
