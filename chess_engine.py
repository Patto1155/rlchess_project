import chess
import numpy as np
import torch

class ChessEngine:
    """
    Chess engine wrapper that handles board representation, move encoding/decoding,
    and provides the interface between python-chess and our neural network.
    """
    
    def __init__(self):
        self.board = chess.Board()
        # Piece mapping for neural network input
        self.piece_to_int = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
    
    def reset(self):
        """Reset board to starting position."""
        self.board = chess.Board()
    
    def copy(self):
        """Create a copy of the current game state."""
        engine_copy = ChessEngine()
        engine_copy.board = self.board.copy()
        return engine_copy
    
    def get_legal_moves(self):
        """Get all legal moves in current position."""
        return list(self.board.legal_moves)
    
    def make_move(self, move):
        """Make a move on the board."""
        if move in self.board.legal_moves:
            self.board.push(move)
            return True
        return False
    
    def undo_move(self):
        """Undo the last move."""
        if self.board.move_stack:
            self.board.pop()
    
    def is_game_over(self):
        """Check if game is over."""
        return self.board.is_game_over()
    
    def get_result(self):
        """
        Get game result from current player's perspective.
        Returns: 1 for win, -1 for loss, 0 for draw
        """
        if not self.is_game_over():
            return None
        
        result = self.board.result()
        if result == "1-0":  # White wins
            return 1 if self.board.turn == chess.WHITE else -1
        elif result == "0-1":  # Black wins
            return 1 if self.board.turn == chess.BLACK else -1
        else:  # Draw
            return 0
    
    def board_to_tensor(self):
        """
        Convert board position to neural network input tensor.
        Returns 8x8x12 tensor (6 piece types × 2 colors).
        """
        tensor = np.zeros((8, 8, 12), dtype=np.float32)
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is not None:
                row = 7 - (square // 8)  # Convert to array coordinates
                col = square % 8
                
                piece_type = self.piece_to_int[piece.piece_type]
                color_offset = 0 if piece.color == chess.WHITE else 6
                
                tensor[row, col, piece_type + color_offset] = 1.0
        
        # If it's black's turn, flip the board perspective
        if self.board.turn == chess.BLACK:
            tensor = np.flip(tensor, axis=0).copy()  # Add .copy() to fix negative stride
            # Swap white and black pieces
            white_pieces = tensor[:, :, :6].copy()
            black_pieces = tensor[:, :, 6:].copy()
            tensor[:, :, :6] = black_pieces
            tensor[:, :, 6:] = white_pieces
        
        return torch.FloatTensor(tensor)
    
    def move_to_index(self, move):
        """
        Convert a chess move to neural network output index.
        Uses from_square * 64 + to_square encoding.
        """
        from_square = move.from_square
        to_square = move.to_square
        
        # Flip squares if black to move (for consistency with board flipping)
        if self.board.turn == chess.BLACK:
            from_square = chess.square_mirror(from_square)
            to_square = chess.square_mirror(to_square)
        
        return from_square * 64 + to_square
    
    def index_to_move(self, index):
        """
        Convert neural network output index back to chess move.
        """
        from_square = index // 64
        to_square = index % 64
        
        # Flip squares if black to move
        if self.board.turn == chess.BLACK:
            from_square = chess.square_mirror(from_square)
            to_square = chess.square_mirror(to_square)
        
        try:
            move = chess.Move(from_square, to_square)
            # Handle promotions (default to queen)
            if (self.board.piece_at(from_square) and 
                self.board.piece_at(from_square).piece_type == chess.PAWN):
                if (chess.square_rank(to_square) == 7 and self.board.turn == chess.WHITE) or \
                   (chess.square_rank(to_square) == 0 and self.board.turn == chess.BLACK):
                    move.promotion = chess.QUEEN
            
            return move if move in self.board.legal_moves else None
        except:
            return None
    
    def get_move_probabilities_mask(self):
        """
        Get a mask for legal moves to apply to neural network policy output.
        Returns array of size 4096 with 1s for legal moves, 0s for illegal.
        """
        mask = np.zeros(4096, dtype=np.float32)
        for move in self.get_legal_moves():
            move_index = self.move_to_index(move)
            mask[move_index] = 1.0
        return mask
    
    def get_fen(self):
        """Get current position as FEN string."""
        return self.board.fen()
    
    def set_fen(self, fen):
        """Set position from FEN string."""
        self.board.set_fen(fen)
    
    def __str__(self):
        """String representation of the board."""
        return str(self.board) 