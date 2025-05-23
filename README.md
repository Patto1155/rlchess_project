# RL Chess - Reinforcement Learning Chess System

A chess AI that learns through self-play using reinforcement learning, starting with only knowledge of chess rules.

## Features
- ♟️ **Self-Play Learning**: Bot learns chess from scratch through AlphaZero-style training
- 👁️ **Live Spectating**: Watch the AI play against itself during training
- 🎮 **Human vs AI**: Play against the trained bot
- 🎨 **Modern UI**: Clean, intuitive interface built with Python/Tkinter
- 🚀 **Ready to Run**: Easy setup with minimal dependencies

## Quick Start

### Installation
```bash
# Clone or download the project
cd rlchess

# Install dependencies
pip install -r requirements.txt
```

### Running the System
```bash
# Start the chess system (training + UI)
python main.py
```

## How It Works

The system implements an AlphaZero-style algorithm:

1. **Neural Network**: Evaluates board positions and suggests moves
2. **Monte Carlo Tree Search**: Explores possible moves using the neural network
3. **Self-Play**: Generates training data by playing games against itself
4. **Continuous Learning**: Updates the neural network based on game outcomes

## Usage

### Training Mode
- The AI starts playing against itself immediately
- Watch live games in the main window
- Training statistics update in real-time
- Model automatically saves progress

### Playing Against the AI
- Click "Play vs AI" to start a human game
- Click squares to select and move pieces
- The AI will respond with its best move
- Game history and analysis available

## Architecture

- `main.py` - Entry point and UI coordination
- `chess_engine.py` - Chess rules and board management
- `neural_network.py` - PyTorch neural network model
- `mcts.py` - Monte Carlo Tree Search implementation
- `trainer.py` - Self-play training loop
- `gui.py` - Graphical user interface

## System Requirements

- Python 3.8+
- 4GB+ RAM recommended
- GPU optional (speeds up training significantly)

## License

MIT License - Feel free to modify and distribute! 