import threading
import time
import random
from collections import deque
import numpy as np

from chess_engine import ChessEngine
from neural_network import ChessNetworkManager
from mcts import MCTSPlayer

class TrainingData:
    """Container for training data from self-play games."""
    
    def __init__(self, board_tensor, policy_target, value_target):
        self.board_tensor = board_tensor
        self.policy_target = policy_target
        self.value_target = value_target

class GameRecord:
    """Record of a complete self-play game with training data."""
    
    def __init__(self):
        self.positions = []  # List of (board_tensor, policy_target) tuples
        self.result = None   # Final game result
        self.move_history = []  # List of moves for display
    
    def add_position(self, board_tensor, policy_target, move):
        self.positions.append((board_tensor, policy_target))
        self.move_history.append(move)
    
    def finalize(self, result):
        self.result = result
    
    def get_training_data(self):
        """Convert game record to training data with proper value targets."""
        training_data = []
        
        for i, (board_tensor, policy_target) in enumerate(self.positions):
            # Calculate value target based on game result and position in game
            # Value decays based on how far the position is from the game end
            moves_from_end = len(self.positions) - i - 1
            decay_factor = 0.99 ** moves_from_end
            
            # Alternate perspective for each move
            if i % 2 == 0:  # White to move
                value_target = self.result * decay_factor
            else:  # Black to move  
                value_target = -self.result * decay_factor
            
            training_data.append(TrainingData(board_tensor, policy_target, value_target))
        
        return training_data

class ChessTrainer:
    """
    Main training system that coordinates self-play games and neural network updates.
    """
    
    def __init__(self, network_manager, buffer_size=50000, batch_size=32, 
                 games_per_iteration=100, training_iterations_per_cycle=10):
        self.network_manager = network_manager
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.games_per_iteration = games_per_iteration
        self.training_iterations_per_cycle = training_iterations_per_cycle
        
        # Training data buffer
        self.training_buffer = deque(maxlen=buffer_size)
        
        # Training statistics
        self.total_games = 0
        self.total_training_steps = 0
        self.current_game_record = None
        
        # Training control
        self.is_training = False
        self.training_thread = None
        
        # Callbacks for UI updates
        self.game_update_callback = None
        self.stats_update_callback = None
        
        print(f"Chess trainer initialized with buffer size {buffer_size}")
    
    def set_callbacks(self, game_update_callback=None, stats_update_callback=None):
        """Set callbacks for UI updates."""
        self.game_update_callback = game_update_callback
        self.stats_update_callback = stats_update_callback
    
    def start_training(self):
        """Start the training process in a separate thread."""
        if self.is_training:
            return
        
        self.is_training = True
        self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.training_thread.start()
        print("Training started")
        
        # Immediately update statistics when training starts
        if self.stats_update_callback:
            stats = self.get_training_stats()
            self.stats_update_callback(stats)
    
    def stop_training(self):
        """Stop the training process."""
        self.is_training = False
        if self.training_thread and self.training_thread.is_alive():
            print("Stopping training...")
            # Give the thread a reasonable time to stop gracefully
            self.training_thread.join(timeout=10.0)
            
            # If thread is still alive after timeout, it's stuck
            if self.training_thread.is_alive():
                print("Warning: Training thread didn't stop cleanly")
            else:
                print("Training stopped")
        else:
            print("Training stopped")
    
    def _training_loop(self):
        """Main training loop that runs in a separate thread."""
        try:
            while self.is_training:
                # Generate self-play games
                for _ in range(self.games_per_iteration):
                    if not self.is_training:
                        break
                    
                    game_record = self._play_self_play_game()
                    if game_record:
                        self._add_game_to_buffer(game_record)
                    
                    # Update statistics after each game for more responsive UI
                    if self.stats_update_callback:
                        try:
                            stats = self.get_training_stats()
                            self.stats_update_callback(stats)
                        except Exception as e:
                            print(f"Stats callback error: {e}")
                
                # Train neural network if we have enough data
                if len(self.training_buffer) >= self.batch_size:
                    for _ in range(self.training_iterations_per_cycle):
                        if not self.is_training:
                            break
                        self._train_network()
                
                # Save model periodically
                if self.total_games % 100 == 0 and self.total_games > 0:
                    self.network_manager.save_model(f"chess_model_games_{self.total_games}.pth")
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)
        
        except Exception as e:
            print(f"Training error: {e}")
            self.is_training = False
    
    def _play_self_play_game(self):
        """Play one self-play game and return the game record."""
        try:
            game_engine = ChessEngine()
            game_record = GameRecord()
            
            # Create players with some randomness for exploration
            temperature = max(0.1, 1.0 - (self.total_games / 1000))  # Decrease over time
            player = MCTSPlayer(self.network_manager, num_simulations=150, temperature=temperature)
            
            move_count = 0
            max_moves = 200  # Prevent infinite games
            
            while not game_engine.is_game_over() and move_count < max_moves:
                # Get move and training data
                move, board_tensor, policy_target = player.get_move_with_policy(game_engine)
                
                if move is None:
                    break
                
                # Record position for training
                game_record.add_position(board_tensor, policy_target, move)
                
                # Make the move
                game_engine.make_move(move)
                move_count += 1
                
                # Update UI if callback is set
                if self.game_update_callback:
                    self.current_game_record = game_record
                    self.game_update_callback(game_engine.copy(), move)
            
            # Finalize game record with result
            result = game_engine.get_result()
            if result is None:
                result = 0  # Draw if max moves reached
            
            game_record.finalize(result)
            self.total_games += 1
            
            return game_record
        
        except Exception as e:
            print(f"Error in self-play game: {e}")
            return None
    
    def _add_game_to_buffer(self, game_record):
        """Add training data from a game to the buffer."""
        training_data = game_record.get_training_data()
        
        for data in training_data:
            self.training_buffer.append(data)
    
    def _train_network(self):
        """Train the neural network on a batch of data."""
        if len(self.training_buffer) < self.batch_size:
            return
        
        # Sample random batch
        batch_data = random.sample(self.training_buffer, self.batch_size)
        
        # Prepare batch tensors
        batch_boards = [data.board_tensor for data in batch_data]
        batch_policies = [data.policy_target for data in batch_data]
        batch_values = [data.value_target for data in batch_data]
        
        # Train the network
        loss_info = self.network_manager.train_step(batch_boards, batch_policies, batch_values)
        
        self.total_training_steps += 1
        
        # Print training progress occasionally
        if self.total_training_steps % 100 == 0:
            print(f"Training step {self.total_training_steps}: Loss = {loss_info['total_loss']:.4f}")
    
    def get_training_stats(self):
        """Get current training statistics."""
        return {
            'total_games': self.total_games,
            'total_training_steps': self.total_training_steps,
            'buffer_size': len(self.training_buffer),
            'is_training': self.is_training,
            'buffer_capacity': self.buffer_size
        }
    
    def get_current_game(self):
        """Get the current game being played for display."""
        return self.current_game_record
    
    def play_human_vs_ai(self, human_is_white=True):
        """
        Create a human vs AI game session.
        Returns a player object configured for human gameplay.
        """
        # Use lower temperature and more simulations for stronger play
        return MCTSPlayer(self.network_manager, num_simulations=800, temperature=0.1)

class QuickPlayer:
    """
    Faster player for quick evaluations and demonstrations.
    Uses fewer MCTS simulations for faster play.
    """
    
    def __init__(self, network_manager, num_simulations=100):
        self.player = MCTSPlayer(network_manager, num_simulations=num_simulations, temperature=0.2)
    
    def select_move(self, game_state):
        return self.player.select_move(game_state) 