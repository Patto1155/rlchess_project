import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import chess
from PIL import Image, ImageTk
import os

from chess_engine import ChessEngine
from trainer import ChessTrainer

class ChessGUI:
    """
    Main GUI application for the RL Chess system.
    Provides live training visualization and human vs AI gameplay.
    """
    
    def __init__(self, trainer):
        self.trainer = trainer
        self.root = tk.Tk()
        self.root.title("RL Chess - Reinforcement Learning Chess System")
        self.root.geometry("1200x800")
        
        # Game state for display
        self.display_engine = ChessEngine()
        self.human_game_engine = None
        self.ai_player = None
        self.human_is_white = True
        self.selected_square = None
        self.in_human_game = False
        
        # UI update control
        self.update_lock = threading.Lock()
        
        # Create UI
        self._create_widgets()
        self._setup_callbacks()
        
        # Start UI update loop
        self._start_ui_updates()
    
    def _create_widgets(self):
        """Create all GUI widgets."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Left panel - Chess board
        self._create_board_panel(main_frame)
        
        # Right panel - Controls and statistics
        self._create_control_panel(main_frame)
    
    def _create_board_panel(self, parent):
        """Create the chess board display panel."""
        board_frame = ttk.LabelFrame(parent, text="Chess Board", padding="10")
        board_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Chess board canvas
        self.board_canvas = tk.Canvas(board_frame, width=480, height=480, bg='white')
        self.board_canvas.grid(row=0, column=0, pady=(0, 10))
        
        # Bind mouse clicks for human moves
        self.board_canvas.bind("<Button-1>", self._on_board_click)
        
        # Game status label
        self.status_label = ttk.Label(board_frame, text="Training Mode - Watching AI vs AI", 
                                     font=("Arial", 12, "bold"))
        self.status_label.grid(row=1, column=0, pady=5)
        
        # Move history
        history_frame = ttk.LabelFrame(board_frame, text="Move History", padding="5")
        history_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.history_text = tk.Text(history_frame, height=8, width=60, wrap=tk.WORD)
        history_scroll = ttk.Scrollbar(history_frame, orient="vertical", command=self.history_text.yview)
        self.history_text.configure(yscrollcommand=history_scroll.set)
        
        self.history_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        history_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        history_frame.columnconfigure(0, weight=1)
        history_frame.rowconfigure(0, weight=1)
        
        # Initialize board
        self._draw_board()
    
    def _create_control_panel(self, parent):
        """Create the control and statistics panel."""
        control_frame = ttk.Frame(parent)
        control_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        control_frame.rowconfigure(1, weight=1)
        
        # Training controls
        training_frame = ttk.LabelFrame(control_frame, text="Training Control", padding="10")
        training_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Fast mode toggle
        self.fast_mode_var = tk.BooleanVar()
        self.fast_mode_checkbox = ttk.Checkbutton(training_frame, text="Fast Mode (3x faster, smaller network)", 
                                                 variable=self.fast_mode_var, command=self._on_fast_mode_toggle)
        self.fast_mode_checkbox.grid(row=0, column=0, columnspan=3, sticky=(tk.W), pady=(0, 10))
        
        # Training buttons
        self.start_button = ttk.Button(training_frame, text="Start Training", 
                                      command=self._start_training)
        self.start_button.grid(row=1, column=0, padx=(0, 5))
        
        self.stop_button = ttk.Button(training_frame, text="Stop Training", 
                                     command=self._stop_training, state="disabled")
        self.stop_button.grid(row=1, column=1, padx=5)
        
        self.save_button = ttk.Button(training_frame, text="Save Model", 
                                     command=self._save_model)
        self.save_button.grid(row=1, column=2, padx=(5, 0))
        
        # Human vs AI controls
        game_frame = ttk.LabelFrame(control_frame, text="Play Against AI", padding="10")
        game_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.play_white_button = ttk.Button(game_frame, text="Play as White", 
                                           command=lambda: self._start_human_game(True))
        self.play_white_button.grid(row=0, column=0, padx=(0, 5))
        
        self.play_black_button = ttk.Button(game_frame, text="Play as Black", 
                                           command=lambda: self._start_human_game(False))
        self.play_black_button.grid(row=0, column=1, padx=5)
        
        self.spectate_button = ttk.Button(game_frame, text="Spectate Training", 
                                         command=self._return_to_spectate)
        self.spectate_button.grid(row=1, column=0, columnspan=2, pady=(10, 0), sticky=(tk.W, tk.E))
        
        # Statistics display
        stats_frame = ttk.LabelFrame(control_frame, text="Training Statistics", padding="10")
        stats_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        stats_frame.rowconfigure(0, weight=1)
        
        self.stats_text = tk.Text(stats_frame, height=15, wrap=tk.WORD, state="disabled")
        stats_scroll = ttk.Scrollbar(stats_frame, orient="vertical", command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scroll.set)
        
        self.stats_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        stats_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        stats_frame.columnconfigure(0, weight=1)
    
    def _setup_callbacks(self):
        """Set up trainer callbacks for UI updates."""
        self.trainer.set_callbacks(
            game_update_callback=self._on_training_game_update,
            stats_update_callback=self._on_stats_update
        )
    
    def _draw_board(self):
        """Draw the chess board and pieces."""
        self.board_canvas.delete("all")
        
        square_size = 60
        
        # Draw squares
        for row in range(8):
            for col in range(8):
                x1 = col * square_size
                y1 = row * square_size
                x2 = x1 + square_size
                y2 = y1 + square_size
                
                # Alternate colors
                if (row + col) % 2 == 0:
                    color = "#f0d9b5"  # Light squares
                else:
                    color = "#b58863"  # Dark squares
                
                # Highlight selected square
                if self.selected_square is not None:
                    selected_row = 7 - (self.selected_square // 8)
                    selected_col = self.selected_square % 8
                    if row == selected_row and col == selected_col:
                        color = "#ffff00"  # Yellow highlight
                
                self.board_canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")
        
        # Draw pieces
        for square in chess.SQUARES:
            piece = self.display_engine.board.piece_at(square)
            if piece is not None:
                row = 7 - (square // 8)
                col = square % 8
                
                x = col * square_size + square_size // 2
                y = row * square_size + square_size // 2
                
                piece_text = self._get_piece_symbol(piece)
                self.board_canvas.create_text(x, y, text=piece_text, font=("Arial", 24), 
                                            fill="black" if piece.color == chess.WHITE else "red")
        
        # Draw coordinates
        for i in range(8):
            # Files (a-h)
            file_char = chr(ord('a') + i)
            self.board_canvas.create_text(i * square_size + square_size // 2, 8 * square_size + 10, 
                                        text=file_char, font=("Arial", 10))
            
            # Ranks (1-8)
            rank_char = str(8 - i)
            self.board_canvas.create_text(-15, i * square_size + square_size // 2, 
                                        text=rank_char, font=("Arial", 10))
    
    def _get_piece_symbol(self, piece):
        """Get Unicode symbol for a chess piece."""
        symbols = {
            (chess.PAWN, chess.WHITE): "♙",
            (chess.ROOK, chess.WHITE): "♖",
            (chess.KNIGHT, chess.WHITE): "♘", 
            (chess.BISHOP, chess.WHITE): "♗",
            (chess.QUEEN, chess.WHITE): "♕",
            (chess.KING, chess.WHITE): "♔",
            (chess.PAWN, chess.BLACK): "♟",
            (chess.ROOK, chess.BLACK): "♜",
            (chess.KNIGHT, chess.BLACK): "♞",
            (chess.BISHOP, chess.BLACK): "♝",
            (chess.QUEEN, chess.BLACK): "♛",
            (chess.KING, chess.BLACK): "♚",
        }
        return symbols.get((piece.piece_type, piece.color), "?")
    
    def _on_board_click(self, event):
        """Handle mouse clicks on the chess board."""
        if not self.in_human_game or self.human_game_engine is None:
            return
        
        square_size = 60
        col = event.x // square_size
        row = event.y // square_size
        
        if 0 <= row < 8 and 0 <= col < 8:
            clicked_square = (7 - row) * 8 + col
            
            if self.selected_square is None:
                # Select a square
                piece = self.human_game_engine.board.piece_at(clicked_square)
                if piece is not None and piece.color == self.human_game_engine.board.turn:
                    if (self.human_is_white and piece.color == chess.WHITE) or \
                       (not self.human_is_white and piece.color == chess.BLACK):
                        self.selected_square = clicked_square
                        self._draw_board()
            else:
                # Try to make a move
                try:
                    move = chess.Move(self.selected_square, clicked_square)
                    
                    # Handle pawn promotion (default to queen)
                    piece = self.human_game_engine.board.piece_at(self.selected_square)
                    if (piece and piece.piece_type == chess.PAWN and 
                        ((chess.square_rank(clicked_square) == 7 and piece.color == chess.WHITE) or
                         (chess.square_rank(clicked_square) == 0 and piece.color == chess.BLACK))):
                        move.promotion = chess.QUEEN
                    
                    if move in self.human_game_engine.get_legal_moves():
                        self._make_human_move(move)
                    else:
                        self.selected_square = None
                        self._draw_board()
                except:
                    self.selected_square = None
                    self._draw_board()
    
    def _make_human_move(self, move):
        """Process a human move and get AI response."""
        self.selected_square = None
        
        # Make human move
        self.human_game_engine.make_move(move)
        self.display_engine = self.human_game_engine.copy()
        self._draw_board()
        self._update_move_history(move)
        
        # Check if game is over
        if self.human_game_engine.is_game_over():
            self._end_human_game()
            return
        
        # Get AI move in separate thread to avoid blocking UI
        threading.Thread(target=self._get_ai_move, daemon=True).start()
    
    def _get_ai_move(self):
        """Get AI move in a separate thread."""
        try:
            ai_move = self.ai_player.select_move(self.human_game_engine)
            
            if ai_move:
                self.human_game_engine.make_move(ai_move)
                
                # Update display in main thread
                self.root.after(0, lambda: self._update_after_ai_move(ai_move))
        except Exception as e:
            print(f"Error getting AI move: {e}")
    
    def _update_after_ai_move(self, ai_move):
        """Update display after AI move (called in main thread)."""
        self.display_engine = self.human_game_engine.copy()
        self._draw_board()
        self._update_move_history(ai_move)
        
        if self.human_game_engine.is_game_over():
            self._end_human_game()
    
    def _on_fast_mode_toggle(self):
        """Handle fast mode checkbox toggle."""
        if self.trainer.is_training:
            # If training is running, show message and revert the checkbox
            current_state = not self.fast_mode_var.get()
            self.fast_mode_var.set(current_state)
            messagebox.showinfo("Fast Mode", 
                              "Please stop training before changing fast mode.\n"
                              "The new setting will apply when you restart training.")
        else:
            # Training is stopped, show confirmation of the change
            mode_text = "Fast Mode enabled" if self.fast_mode_var.get() else "Full strength mode enabled"
            print(f"🔧 {mode_text}")
    
    def _start_training(self):
        """Start the training process."""
        # Check if we need to reinitialize with different fast mode setting
        current_fast_mode = getattr(self.trainer.network_manager, 'fast_mode', False)
        new_fast_mode = self.fast_mode_var.get()
        
        if current_fast_mode != new_fast_mode:
            # Need to create new network manager with different architecture
            print(f"🔄 Switching to {'fast' if new_fast_mode else 'full'} mode...")
            
            # Import here to avoid circular imports
            from neural_network import ChessNetworkManager
            from trainer import ChessTrainer
            
            # Create new network manager with the selected mode
            new_network_manager = ChessNetworkManager(
                device=self.trainer.network_manager.device, 
                fast_mode=new_fast_mode
            )
            
            # Create new trainer with the new network
            self.trainer = ChessTrainer(
                network_manager=new_network_manager,
                buffer_size=self.trainer.buffer_size,
                batch_size=self.trainer.batch_size,
                games_per_iteration=self.trainer.games_per_iteration
            )
            
            # Reconnect callbacks
            self._setup_callbacks()
        
        self.trainer.start_training()
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.fast_mode_checkbox.config(state="disabled")  # Disable toggle during training
    
    def _stop_training(self):
        """Stop the training process."""
        try:
            self.trainer.stop_training()
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.fast_mode_checkbox.config(state="normal")  # Re-enable toggle when stopped
        except Exception as e:
            print(f"Error stopping training: {e}")
            # Reset button states even if there was an error
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.fast_mode_checkbox.config(state="normal")
    
    def _save_model(self):
        """Save the current model."""
        self.trainer.network_manager.save_model()
        messagebox.showinfo("Model Saved", "Neural network model has been saved successfully!")
    
    def _start_human_game(self, human_is_white):
        """Start a human vs AI game."""
        self.human_is_white = human_is_white
        self.human_game_engine = ChessEngine()
        self.display_engine = self.human_game_engine.copy()
        self.ai_player = self.trainer.play_human_vs_ai(human_is_white)
        self.in_human_game = True
        
        self.status_label.config(text=f"Human vs AI - You are {'White' if human_is_white else 'Black'}")
        self._clear_move_history()
        self._draw_board()
        
        # If human is black, get AI's first move
        if not human_is_white:
            threading.Thread(target=self._get_ai_move, daemon=True).start()
    
    def _return_to_spectate(self):
        """Return to spectating training mode."""
        self.in_human_game = False
        self.human_game_engine = None
        self.ai_player = None
        self.selected_square = None
        
        self.status_label.config(text="Training Mode - Watching AI vs AI")
        self._draw_board()
    
    def _end_human_game(self):
        """Handle end of human vs AI game."""
        result = self.human_game_engine.get_result()
        if result == 1:
            if self.human_is_white:
                message = "Congratulations! You won!"
            else:
                message = "AI wins! Better luck next time."
        elif result == -1:
            if self.human_is_white:
                message = "AI wins! Better luck next time."
            else:
                message = "Congratulations! You won!"
        else:
            message = "Game ended in a draw."
        
        messagebox.showinfo("Game Over", message)
        self._return_to_spectate()
    
    def _on_training_game_update(self, game_engine, move):
        """Callback for training game updates."""
        if not self.in_human_game:
            with self.update_lock:
                self.display_engine = game_engine
            # Schedule UI update in main thread
            self.root.after(0, self._update_display_for_training)
    
    def _update_display_for_training(self):
        """Update display for training game (called in main thread)."""
        if not self.in_human_game:
            self._draw_board()
    
    def _on_stats_update(self, stats):
        """Callback for statistics updates."""
        # Schedule stats update in main thread
        self.root.after(0, lambda: self._update_stats_display(stats))
    
    def _update_stats_display(self, stats):
        """Update statistics display (called in main thread)."""
        try:
            self.stats_text.config(state="normal")
            self.stats_text.delete(1.0, tk.END)
            
            stats_text = f"""Training Statistics:

Games Played: {stats['total_games']}
Training Steps: {stats['total_training_steps']}
Buffer Size: {stats['buffer_size']} / {stats['buffer_capacity']}
Status: {'Training' if stats['is_training'] else 'Stopped'}

Model Information:
Device: {self.trainer.network_manager.device}
Parameters: {self.trainer.network_manager.get_model_info()['total_parameters']:,}

Performance:
- Games per minute: {stats['total_games'] / max(1, time.time() - getattr(self, '_start_time', time.time())) * 60:.1f}
- Training efficiency: {'Good' if stats['buffer_size'] > 1000 else 'Building up data...'}

Tips:
- Let the system train for at least 100 games before playing
- Training improves with time - be patient!
- Use 'Save Model' to preserve progress
"""
            
            self.stats_text.insert(1.0, stats_text)
            self.stats_text.config(state="disabled")
            
        except Exception as e:
            print(f"Error updating stats display: {e}")
            # Insert basic error message so user knows something is happening
            self.stats_text.config(state="normal")
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, f"Training running... (display error: {e})")
            self.stats_text.config(state="disabled")
    
    def _update_move_history(self, move):
        """Add move to the move history display."""
        self.history_text.insert(tk.END, f"{move} ")
        self.history_text.see(tk.END)
    
    def _clear_move_history(self):
        """Clear the move history display."""
        self.history_text.delete(1.0, tk.END)
    
    def _start_ui_updates(self):
        """Start the UI update loop."""
        self._start_time = time.time()
        self._ui_update_loop()
    
    def _ui_update_loop(self):
        """Periodic UI updates."""
        # Schedule next update
        self.root.after(1000, self._ui_update_loop)  # Update every second
    
    def run(self):
        """Start the GUI application."""
        print("Starting RL Chess GUI...")
        self.root.mainloop()
    
    def cleanup(self):
        """Clean up resources when closing."""
        self.trainer.stop_training() 