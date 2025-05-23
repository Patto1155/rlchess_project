# Technical Context

## Technology Stack

### Core Libraries
- **PyTorch**: Neural network implementation and training
- **python-chess**: Chess rules, board representation, move validation
- **tkinter**: GUI framework (built into Python, cross-platform)
- **numpy**: Numerical computations
- **threading**: Concurrent training and UI updates

### Algorithm Choice: AlphaZero-Style
- **Monte Carlo Tree Search (MCTS)**: Game tree exploration
- **Neural Network**: Position evaluation + move probability prediction
- **Self-Play Training**: Generate training data from bot vs bot games
- **Policy + Value Network**: Single network predicting both move probabilities and position values

### Architecture Components

#### 1. Chess Engine (`chess_engine.py`)
- Uses `python-chess` library for rules
- Board state representation
- Legal move generation
- Game termination detection

#### 2. Neural Network (`neural_network.py`)
- Input: 8x8x12 board representation (piece positions)
- Output: Move probabilities (4096 possible moves) + Position value (-1 to 1)
- Convolutional layers for spatial pattern recognition

#### 3. MCTS (`mcts.py`)
- Tree search with neural network guidance
- UCB1 selection for exploration vs exploitation
- Simulation rollouts using neural network

#### 4. Training System (`trainer.py`)
- Self-play game generation
- Experience buffer management
- Neural network training loop
- Model checkpointing

#### 5. UI System (`gui.py`)
- Real-time board visualization
- Training statistics display
- Human vs bot gameplay interface
- Move history and analysis

### Development Setup
- Python 3.8+
- Virtual environment recommended
- GPU support optional but recommended for training speed 