import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import chess.svg
import tkinter as tk
from io import BytesIO
from PIL import Image, ImageTk
import cairosvg

### ============================
### 1️⃣ Residual Block (Fixes Model Compatibility)
### ============================
class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.bn2 = nn.BatchNorm2d(hidden_size)
        self.activation1 = nn.SELU()
        self.activation2 = nn.SELU()

    def forward(self, x):
        x_input = torch.clone(x)  # Residual connection
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + x_input  # Skip connection
        x = self.activation2(x)
        return x

### ============================
### 2️⃣ Chess AI Model (Fixes Naming Issues)
### ============================
class ChessNet(nn.Module):
    def __init__(self, hidden_layers=4, hidden_size=200):
        super(ChessNet, self).__init__()
        self.hidden_layers = hidden_layers
        self.input_layer = nn.Conv2d(6, hidden_size, 3, stride=1, padding=1)

        # Renamed module_list to residual_blocks (Matches trained model)
        self.residual_blocks = nn.ModuleList([ResidualBlock(hidden_size) for _ in range(hidden_layers)])

        self.output_layer = nn.Conv2d(hidden_size, 2, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        for block in self.residual_blocks:  # Matches trained model structure
            x = block(x)
        x = self.output_layer(x)
        return x

### ============================
### 3️⃣ Load Model (Fixes DataParallel Issues)
### ============================
model = ChessNet()

# Load state dict and remove 'module.' prefix if needed
state_dict = torch.load("chessnet_model.pth")

# Handle DataParallel-saved models
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
try:
    model.load_state_dict(new_state_dict, strict=True)
except RuntimeError as e:
    print(f"Warning: Loading with strict=False due to mismatch: {e}")
    model.load_state_dict(new_state_dict, strict=False)

model.eval()

### ============================
### 4️⃣ Helper Functions
### ============================
def board_2_rep(board):
    """ Convert chess board to tensor representation """
    piece_map = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5, 'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}
    board_matrix = torch.zeros((6, 8, 8), dtype=torch.float32)
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)
            idx = piece_map[str(piece)] % 6
            board_matrix[idx, row, col] = 1 if piece.color == chess.WHITE else -1

    return board_matrix.unsqueeze(0)  # Add batch dimension

def rep_to_move(y_pred, board):
    """ Convert model output into a legal move """
    legal_moves = list(board.legal_moves)
    move_strs = [move.uci() for move in legal_moves]
    return move_strs[torch.randint(len(move_strs), (1,)).item()]  # Pick a random valid move (temporary)

### ============================
### 5️⃣ GUI Chess App
### ============================
class ChessGUI:
    def __init__(self, root):
        self.root = root
        self.board = chess.Board()
        
        self.canvas = tk.Canvas(root, width=400, height=400)
        self.canvas.pack()

        self.move_entry = tk.Entry(root)
        self.move_entry.pack()

        self.move_button = tk.Button(root, text="Make Move", command=self.make_move)
        self.move_button.pack()

        self.ai_button = tk.Button(root, text="AI Move", command=self.ai_move)
        self.ai_button.pack()

        self.status_label = tk.Label(root, text="Your Turn!")
        self.status_label.pack()

        self.update_board()

    def update_board(self):
        """ Convert SVG chessboard to a PNG and display in Tkinter """
        board_svg = chess.svg.board(self.board)
        
        # Convert SVG to PNG using cairosvg
        png_data = cairosvg.svg2png(bytestring=board_svg.encode('utf-8'))

        # Open PNG with PIL
        img = Image.open(BytesIO(png_data))
        img = img.resize((400, 400), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)


    def make_move(self):
        """ Handle player's move """
        move = self.move_entry.get()
        if move in [m.uci() for m in self.board.legal_moves]:
            self.board.push_uci(move)
            self.status_label.config(text="AI Thinking...")
            self.update_board()
        else:
            self.status_label.config(text="Invalid Move. Try again.")

    def ai_move(self):
        """ AI makes a move using the model """
        if self.board.is_game_over():
            self.status_label.config(text="Game Over!")
            return
        
        x = board_2_rep(self.board)
        with torch.no_grad():
            y_pred = model(x)
        move_pred = rep_to_move(y_pred, self.board)

        self.board.push_uci(move_pred)
        self.status_label.config(text="Your Turn!")
        self.update_board()

### ============================
### 6️⃣ Run GUI Chess App
### ============================
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Chess AI")
    app = ChessGUI(root)
    root.mainloop()
