import math
import random
import numpy as np
from chess_engine import ChessEngine

class MCTSNode:
    """
    Node in the Monte Carlo Tree Search tree.
    Stores game state, visit statistics, and children nodes.
    """
    
    def __init__(self, game_state, parent=None, move=None, prior_prob=0.0):
        self.game_state = game_state  # ChessEngine instance
        self.parent = parent
        self.move = move  # Move that led to this state
        self.prior_prob = prior_prob  # Prior probability from neural network
        
        # MCTS statistics
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}  # Dict mapping moves to child nodes
        
        # Neural network evaluation cache
        self.is_expanded = False
        self.policy_probs = None
        self.value_estimate = None
    
    def is_leaf(self):
        """Check if this is a leaf node (no children)."""
        return len(self.children) == 0
    
    def is_terminal(self):
        """Check if this represents a terminal game state."""
        return self.game_state.is_game_over()
    
    def get_ucb_score(self, c_puct=1.0):
        """
        Calculate UCB1 score for node selection.
        Higher scores indicate nodes that should be explored.
        """
        if self.visit_count == 0:
            return float('inf')
        
        # UCB1 formula with prior probability
        exploitation = self.value_sum / self.visit_count
        exploration = c_puct * self.prior_prob * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        
        return exploitation + exploration
    
    def select_child(self, c_puct=1.0):
        """Select child with highest UCB score."""
        if not self.children:
            return None
        
        return max(self.children.values(), key=lambda child: child.get_ucb_score(c_puct))
    
    def expand(self, network_manager):
        """
        Expand node by adding children for all legal moves.
        Uses neural network to get prior probabilities.
        """
        if self.is_expanded or self.is_terminal():
            return
        
        # Get neural network evaluation
        board_tensor = self.game_state.board_to_tensor()
        policy_probs, value_estimate = network_manager.predict(board_tensor)
        
        self.policy_probs = policy_probs
        self.value_estimate = value_estimate
        
        # Get legal moves and their probabilities
        legal_moves = self.game_state.get_legal_moves()
        move_probs = []
        
        for move in legal_moves:
            move_index = self.game_state.move_to_index(move)
            prob = policy_probs[move_index]
            move_probs.append((move, prob))
        
        # Normalize probabilities for legal moves only
        total_prob = sum(prob for _, prob in move_probs)
        if total_prob > 0:
            move_probs = [(move, prob / total_prob) for move, prob in move_probs]
        else:
            # Uniform distribution if all probabilities are zero
            uniform_prob = 1.0 / len(legal_moves)
            move_probs = [(move, uniform_prob) for move, _ in move_probs]
        
        # Create child nodes
        for move, prob in move_probs:
            child_state = self.game_state.copy()
            child_state.make_move(move)
            
            child_node = MCTSNode(child_state, parent=self, move=move, prior_prob=prob)
            self.children[move] = child_node
        
        self.is_expanded = True
    
    def backup(self, value):
        """
        Backup value through the tree to the root.
        Updates visit counts and value sums.
        """
        self.visit_count += 1
        self.value_sum += value
        
        if self.parent:
            # Flip value for opponent's perspective
            self.parent.backup(-value)
    
    def get_visit_distribution(self, temperature=1.0):
        """
        Get probability distribution over moves based on visit counts.
        Temperature controls exploration (0 = greedy, higher = more random).
        """
        if not self.children:
            return {}
        
        if temperature == 0:
            # Greedy selection
            best_move = max(self.children.keys(), 
                          key=lambda move: self.children[move].visit_count)
            return {move: 1.0 if move == best_move else 0.0 
                   for move in self.children.keys()}
        else:
            # Temperature-based selection
            visit_counts = [self.children[move].visit_count for move in self.children.keys()]
            
            # Apply temperature
            if temperature != 1.0:
                visit_counts = [count ** (1.0 / temperature) for count in visit_counts]
            
            # Normalize
            total = sum(visit_counts)
            if total == 0:
                # Uniform if no visits
                probs = [1.0 / len(visit_counts)] * len(visit_counts)
            else:
                probs = [count / total for count in visit_counts]
            
            return {move: prob for move, prob in zip(self.children.keys(), probs)}

class MCTS:
    """
    Monte Carlo Tree Search algorithm for chess move selection.
    Integrates with neural network for position evaluation and move priors.
    """
    
    def __init__(self, network_manager, c_puct=1.0, num_simulations=800):
        self.network_manager = network_manager
        self.c_puct = c_puct  # Exploration constant
        self.num_simulations = num_simulations
    
    def search(self, game_state, temperature=1.0):
        """
        Run MCTS to find the best move in the given position.
        
        Args:
            game_state: ChessEngine representing current position
            temperature: Controls randomness in move selection
            
        Returns:
            best_move: Selected chess move
            move_probs: Probability distribution over all moves
        """
        root = MCTSNode(game_state.copy())
        
        # Run simulations
        for _ in range(self.num_simulations):
            self._simulate(root)
        
        # Get move probabilities based on visit counts
        move_probs = root.get_visit_distribution(temperature)
        
        if not move_probs:
            # No legal moves (shouldn't happen in normal play)
            return None, {}
        
        # Select move based on probabilities
        if temperature == 0:
            # Greedy selection
            best_move = max(move_probs.keys(), key=lambda move: move_probs[move])
        else:
            # Sample from distribution
            moves = list(move_probs.keys())
            probs = list(move_probs.values())
            best_move = np.random.choice(moves, p=probs)
        
        return best_move, move_probs
    
    def _simulate(self, root):
        """
        Run one MCTS simulation from root to leaf and backup the result.
        """
        node = root
        
        # Selection: Navigate to a leaf node
        while not node.is_leaf() and not node.is_terminal():
            node = node.select_child(self.c_puct)
        
        # Expansion: Add children if not terminal
        if not node.is_terminal():
            node.expand(self.network_manager)
            
            # If node has children, select one for evaluation
            if node.children:
                node = node.select_child(self.c_puct)
        
        # Evaluation: Get value estimate
        if node.is_terminal():
            # Use actual game result
            value = node.game_state.get_result()
            if value is None:
                value = 0  # Shouldn't happen, but safety check
        else:
            # Use neural network evaluation
            if node.value_estimate is None:
                board_tensor = node.game_state.board_to_tensor()
                _, value = self.network_manager.predict(board_tensor)
                node.value_estimate = value
            else:
                value = node.value_estimate
        
        # Backup: Propagate value up the tree
        node.backup(value)
    
    def get_training_data(self, game_state, move_probs):
        """
        Extract training data from MCTS search.
        
        Args:
            game_state: ChessEngine representing current position
            move_probs: Probability distribution from MCTS
            
        Returns:
            board_tensor: Neural network input
            policy_target: Target policy distribution
        """
        board_tensor = game_state.board_to_tensor()
        
        # Convert move probabilities to full policy vector
        policy_target = np.zeros(4096, dtype=np.float32)
        for move, prob in move_probs.items():
            move_index = game_state.move_to_index(move)
            policy_target[move_index] = prob
        
        return board_tensor, policy_target

class MCTSPlayer:
    """
    Chess player that uses MCTS for move selection.
    """
    
    def __init__(self, network_manager, num_simulations=800, temperature=1.0):
        self.mcts = MCTS(network_manager, num_simulations=num_simulations)
        self.temperature = temperature
    
    def select_move(self, game_state):
        """Select a move using MCTS."""
        move, move_probs = self.mcts.search(game_state, self.temperature)
        return move
    
    def get_move_with_policy(self, game_state):
        """Get move and policy for training data collection."""
        move, move_probs = self.mcts.search(game_state, self.temperature)
        board_tensor, policy_target = self.mcts.get_training_data(game_state, move_probs)
        return move, board_tensor, policy_target
    
    def set_temperature(self, temperature):
        """Update temperature for move selection."""
        self.temperature = temperature 