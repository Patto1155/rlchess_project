# System Patterns - RL Chess

## System Architecture
```mermaid
graph TD
    A[User Interface (gui.py)] --> B(Training Coordinator (trainer.py));
    B --> C{Neural Network (neural_network.py)};
    B --> D(MCTS Player (mcts.py));
    D --> C;
    D --> E[Chess Engine (chess_engine.py)];
    A --> E;
    F[Main (main.py)] --> A;
    F --> B;
```

- **main.py**: Entry point, argument parsing, initializes and coordinates major components.
- **gui.py**: Tkinter-based graphical user interface. Handles user interactions for training control and gameplay. Displays board, game history, and statistics.
- **trainer.py**: Manages the self-play training loop. Generates game data, stores it in a buffer, and periodically trains the neural network.
- **neural_network.py**: Defines the PyTorch neural network (ChessNet) and a manager class (ChessNetworkManager) for training, prediction, saving, and loading models.
- **mcts.py**: Implements the Monte Carlo Tree Search algorithm (MCTSNode, MCTS) and an MCTS-based player (MCTSPlayer).
- **chess_engine.py**: Wrapper around the `python-chess` library. Handles board representation, move validation, conversion to tensor format for the NN, and move encoding/decoding.

## Key Technical Decisions
- **AlphaZero-style RL**: The core learning algorithm combines MCTS for strong move selection during self-play and a deep neural network for position evaluation and policy prediction.
- **Self-Play for Data Generation**: The agent learns by playing games against itself, generating board states, move policies (from MCTS), and game outcomes as training data.
- **Replay Buffer**: A deque is used to store recent game positions and outcomes, from which batches are sampled for training the neural network.
- **Dual-headed Neural Network**: The network has two outputs: a policy head (predicting move probabilities) and a value head (evaluating the current board position).
- **Convolutional Neural Network (CNN)**: The network uses convolutional layers to process the 2D board representation, capturing spatial patterns.
- **Threading for UI Responsiveness**: The training loop runs in a separate thread to keep the GUI responsive.

## Design Patterns
- **Model-View-Controller (MVC) variant**: `gui.py` acts as the View, `trainer.py` and `main.py` as Controllers, and the other modules (`chess_engine.py`, `neural_network.py`, `mcts.py`) as the Model.
- **Callbacks**: Used for communication between the `trainer` and `gui` to update the UI with game progress and statistics.
- **Singleton-like Manager**: `ChessNetworkManager` acts as a central point for managing the neural network model.

## Component Relationships
- The `Trainer` uses `MCTSPlayer` to generate self-play games.
- `MCTSPlayer` uses the `ChessNetworkManager` to get predictions from the neural network to guide its search.
- `MCTSPlayer` interacts with `ChessEngine` to make moves and get game state.
- The `Trainer` collects data from `MCTSPlayer` and `ChessEngine` to train the `ChessNetworkManager`.
- The `GUI` interacts with the `Trainer` to start/stop training and to get an `MCTSPlayer` instance for human vs. AI games.
- The `GUI` uses `ChessEngine` to display the board and manage game state for human play.

## Critical Implementation Paths
- **Self-Play Loop**: `Trainer._training_loop` -> `_play_self_play_game` -> `MCTSPlayer.get_move_with_policy` -> `MCTS.search` -> `MCTSNode.expand` (calls NN) / `_simulate`.
- **Network Training**: `Trainer._train_network` -> `ChessNetworkManager.train_step`.
- **Human vs AI**: `GUI._start_human_game` -> `Trainer.play_human_vs_ai` (creates MCTSPlayer) -> `GUI._on_board_click` -> `GUI._make_human_move` -> `MCTSPlayer.select_move` (for AI's turn).
