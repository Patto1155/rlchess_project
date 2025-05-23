#!/usr/bin/env python3
"""
Test script to verify the RL Chess system components work correctly.
Run this to test the system before using the full GUI.
"""

import sys
import time
import chess

def test_chess_engine():
    """Test the chess engine component."""
    print("Testing Chess Engine...")
    
    from chess_engine import ChessEngine
    
    engine = ChessEngine()
    
    # Test basic functionality
    assert not engine.is_game_over()
    assert len(engine.get_legal_moves()) == 20  # Starting position has 20 legal moves
    
    # Test move making
    move = chess.Move.from_uci("e2e4")
    assert engine.make_move(move)
    
    # Test tensor conversion
    tensor = engine.board_to_tensor()
    assert tensor.shape == (8, 8, 12)
    
    # Test move encoding
    move_index = engine.move_to_index(chess.Move.from_uci("e7e5"))
    reconstructed_move = engine.index_to_move(move_index)
    assert reconstructed_move == chess.Move.from_uci("e7e5")
    
    print("✅ Chess Engine tests passed!")

def test_neural_network():
    """Test the neural network component."""
    print("Testing Neural Network...")
    
    from neural_network import ChessNetworkManager
    from chess_engine import ChessEngine
    
    network = ChessNetworkManager()
    engine = ChessEngine()
    
    # Test prediction
    board_tensor = engine.board_to_tensor()
    policy_probs, value = network.predict(board_tensor)
    
    assert len(policy_probs) == 4096
    assert -1 <= value <= 1
    assert abs(sum(policy_probs) - 1.0) < 0.01  # Should sum to ~1
    
    print("✅ Neural Network tests passed!")

def test_mcts():
    """Test the MCTS component."""
    print("Testing MCTS...")
    
    from neural_network import ChessNetworkManager
    from mcts import MCTSPlayer
    from chess_engine import ChessEngine
    
    network = ChessNetworkManager()
    player = MCTSPlayer(network, num_simulations=50, temperature=1.0)
    engine = ChessEngine()
    
    # Test move selection
    move = player.select_move(engine)
    assert move is not None
    assert move in engine.get_legal_moves()
    
    # Test training data generation
    move, board_tensor, policy_target = player.get_move_with_policy(engine)
    assert move is not None
    assert board_tensor.shape == (8, 8, 12)
    assert len(policy_target) == 4096
    assert abs(sum(policy_target) - 1.0) < 0.01
    
    print("✅ MCTS tests passed!")

def test_trainer():
    """Test the trainer component."""
    print("Testing Trainer...")
    
    from neural_network import ChessNetworkManager
    from trainer import ChessTrainer
    
    network = ChessNetworkManager()
    trainer = ChessTrainer(network, buffer_size=1000, batch_size=8, 
                          games_per_iteration=2, training_iterations_per_cycle=1)
    
    # Test statistics
    stats = trainer.get_training_stats()
    assert 'total_games' in stats
    assert 'is_training' in stats
    assert stats['total_games'] == 0
    assert not stats['is_training']
    
    print("✅ Trainer tests passed!")

def test_integration():
    """Test integration between components."""
    print("Testing Component Integration...")
    
    from neural_network import ChessNetworkManager
    from trainer import ChessTrainer
    from chess_engine import ChessEngine
    
    # Initialize components
    network = ChessNetworkManager()
    trainer = ChessTrainer(network, buffer_size=100, batch_size=4, 
                          games_per_iteration=1, training_iterations_per_cycle=1)
    
    # Test a few training iterations
    initial_stats = trainer.get_training_stats()
    
    # Start training briefly
    trainer.start_training()
    time.sleep(2)  # Let it run for 2 seconds
    trainer.stop_training()
    
    final_stats = trainer.get_training_stats()
    
    # Should have made some progress
    assert final_stats['total_games'] >= initial_stats['total_games']
    
    print("✅ Integration tests passed!")

def main():
    """Run all tests."""
    print("=" * 50)
    print("🧪 RL Chess System Tests")
    print("=" * 50)
    print()
    
    try:
        test_chess_engine()
        test_neural_network()
        test_mcts()
        test_trainer()
        test_integration()
        
        print()
        print("=" * 50)
        print("✅ All tests passed! System is ready to use.")
        print("=" * 50)
        print()
        print("To start the full system, run:")
        print("  python main.py")
        print()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 