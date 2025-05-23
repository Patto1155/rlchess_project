#!/usr/bin/env python3
"""
RL Chess - Reinforcement Learning Chess System
Main entry point for the application.

This system implements an AlphaZero-style chess AI that learns through self-play
and provides a GUI for both spectating training and playing against the AI.
"""

import sys
import os
import argparse
import torch

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neural_network import ChessNetworkManager
from trainer import ChessTrainer
from gui import ChessGUI

def main():
    """Main application entry point."""
    
    parser = argparse.ArgumentParser(description="RL Chess - Reinforcement Learning Chess System")
    parser.add_argument("--load-model", type=str, help="Load a previously saved model")
    parser.add_argument("--no-gui", action="store_true", help="Run without GUI (training only)")
    parser.add_argument("--fast-training", action="store_true", help="Use faster training settings (less accurate)")
    parser.add_argument("--fast-mode", action="store_true", help="Use smaller network for faster training (still strong)")
    parser.add_argument("--device", type=str, help="Device to use (cpu/cuda)", default=None)
    
    args = parser.parse_args()
    
    if args.device != "cpu":
        torch.backends.cudnn.benchmark = True

    print("=" * 60)
    print("🏆 RL Chess - Reinforcement Learning Chess System")
    print("=" * 60)
    print()
    print("This system learns chess from scratch using AlphaZero-style")
    print("reinforcement learning with Monte Carlo Tree Search.")
    print()
    
    try:
        # Initialize neural network
        print("🧠 Initializing neural network...")
        network_manager = ChessNetworkManager(device=args.device, fast_mode=args.fast_mode)
        
        # Load existing model if specified
        if args.load_model:
            print(f"📁 Loading model from {args.load_model}...")
            if network_manager.load_model(args.load_model):
                print("✅ Model loaded successfully!")
            else:
                print("❌ Failed to load model, starting with random weights")
        else:
            print("🎲 Starting with randomly initialized network")
        
        # Set training parameters
        if args.fast_training:
            print("⚡ Using fast training settings")
            buffer_size = 10000
            batch_size = 16
            games_per_iteration = 10
        else:
            print("🎯 Using standard training settings")
            buffer_size = 50000
            batch_size = 32
            games_per_iteration = 5  # Reduced for better UI responsiveness
        
        # Initialize trainer
        print("🏋️ Setting up training system...")
        trainer = ChessTrainer(
            network_manager=network_manager,
            buffer_size=buffer_size,
            batch_size=batch_size,
            games_per_iteration=games_per_iteration
        )
        
        if args.no_gui:
            # Run training without GUI
            print("🚀 Starting training (no GUI mode)")
            print("Press Ctrl+C to stop training")
            
            trainer.start_training()
            
            try:
                while True:
                    stats = trainer.get_training_stats()
                    print(f"Games: {stats['total_games']}, "
                          f"Steps: {stats['total_training_steps']}, "
                          f"Buffer: {stats['buffer_size']}")
                    
                    import time
                    time.sleep(10)
                    
            except KeyboardInterrupt:
                print("\n⏹️ Stopping training...")
                trainer.stop_training()
                trainer.network_manager.save_model("final_model.pth")
                print("✅ Model saved. Goodbye!")
        
        else:
            # Run with GUI
            print("🖥️ Starting GUI application...")
            print()
            print("Features:")
            print("- 👁️ Live spectating of AI vs AI training games")
            print("- 🎮 Play against the AI after it has trained")
            print("- 📊 Real-time training statistics")
            print("- 💾 Save and load trained models")
            print()
            print("Usage Tips:")
            print("1. Click 'Start Training' to begin self-play learning")
            print("2. Watch the AI learn by observing the live games")
            print("3. After 50+ games, try playing against the AI")
            print("4. Save your model to preserve progress")
            print()
            print("Speed Options:")
            print("- Use --fast-mode for 3x faster training (smaller network)")
            print("- Human vs AI always uses full strength regardless of mode")
            print()
            
            # Initialize and run GUI
            gui = ChessGUI(trainer)
            
            def cleanup():
                print("🧹 Cleaning up...")
                gui.cleanup()
                trainer.network_manager.save_model("autosave_model.pth")
                print("✅ Cleanup complete")
            
            # Set up cleanup on exit
            import atexit
            atexit.register(cleanup)
            
            try:
                gui.run()
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
    
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 