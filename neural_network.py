import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class ChessNet(nn.Module):
    """
    Neural network for chess position evaluation and move prediction.
    Architecture inspired by AlphaZero with convolutional layers for spatial
    pattern recognition and dual heads for policy and value prediction.
    """
    
    def __init__(self, num_filters=256, num_residual_blocks=10, fast_mode=False):
        super(ChessNet, self).__init__()
        
        # Fast mode uses smaller architecture for speed
        if fast_mode:
            num_filters = 128
            num_residual_blocks = 5
            print("🚀 Using fast mode network (5 blocks, 128 filters)")
        else:
            print("🎯 Using full strength network (10 blocks, 256 filters)")
        
        # Input layer - converts 8x8x12 board to feature maps
        self.input_layer = nn.Conv2d(12, num_filters, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm2d(num_filters)
        
        # Residual blocks for deep feature extraction
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_residual_blocks)
        ])
        
        # Policy head - predicts move probabilities
        self.policy_conv = nn.Conv2d(num_filters, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 4096)  # 64x64 possible moves
        
        # Value head - predicts position evaluation
        self.value_conv = nn.Conv2d(num_filters, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        # Input processing
        x = F.relu(self.input_bn(self.input_layer(x)))
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.reshape(policy.size(0), -1)  # Flatten with reshape instead of view
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.reshape(value.size(0), -1)  # Flatten with reshape instead of view
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # Output between -1 and 1
        
        return policy, value

class ResidualBlock(nn.Module):
    """
    Residual block with skip connections for deep network training.
    """
    
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual  # Skip connection
        out = F.relu(out)
        
        return out

class ChessNetworkManager:
    """
    Manager class for the chess neural network with training, saving, and loading capabilities.
    """
    
    def __init__(self, device=None, model_path="models", fast_mode=False):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.fast_mode = fast_mode
        self.network = ChessNet(fast_mode=fast_mode).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001, weight_decay=1e-4)
        # Initialize GradScaler for AMP if on CUDA
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == 'cuda')
        
        # Create models directory
        os.makedirs(model_path, exist_ok=True)
        
        print(f"Chess network initialized on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.network.parameters()):,}")
    
    def predict(self, board_tensor):
        """
        Predict policy and value for a board position.
        
        Args:
            board_tensor: 8x8x12 tensor representing board state
            
        Returns:
            policy_probs: Probabilities for each possible move
            value: Position evaluation (-1 to 1)
        """
        self.network.eval()
        with torch.no_grad():
            # Add batch dimension if needed
            if len(board_tensor.shape) == 3:
                board_tensor = board_tensor.unsqueeze(0)
            
            # Move to correct device and format (batch, channels, height, width)
            board_tensor = board_tensor.permute(0, 3, 1, 2).to(self.device)
            
            # Use autocast for prediction if on CUDA
            with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
                policy_logits, value = self.network(board_tensor)
            policy_probs = torch.exp(policy_logits)  # Convert from log probabilities
            
            return policy_probs.cpu().numpy()[0], value.cpu().numpy()[0][0]
    
    def train_step(self, batch_boards, batch_policies, batch_values):
        """
        Perform one training step with a batch of data.
        
        Args:
            batch_boards: Batch of board tensors
            batch_policies: Target policy distributions
            batch_values: Target position values
            
        Returns:
            loss: Training loss for this batch
        """
        self.network.train()
        
        # Prepare tensors
        boards = torch.stack(batch_boards).permute(0, 3, 1, 2).to(self.device)
        target_policies = torch.FloatTensor(batch_policies).to(self.device)
        target_values = torch.FloatTensor(batch_values).unsqueeze(1).to(self.device)
        
        # Use autocast for the forward pass if on CUDA
        with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
            pred_policies, pred_values = self.network(boards)
            # Calculate losses
            policy_loss = F.kl_div(pred_policies, target_policies, reduction='batchmean')
            value_loss = F.mse_loss(pred_values, target_values)
            total_loss = policy_loss + value_loss
        
        # Backward pass using GradScaler
        self.optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }
    
    def save_model(self, filename="chess_model.pth"):
        """Save the current model state."""
        filepath = os.path.join(self.model_path, filename)
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() # Save scaler state
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filename="chess_model.pth"):
        """Load a previously saved model state."""
        filepath = os.path.join(self.model_path, filename)
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scaler_state_dict' in checkpoint and self.device.type == 'cuda':
                 self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print(f"Model loaded from {filepath}")
            return True
        else:
            print(f"No model found at {filepath}")
            return False
    
    def get_model_info(self):
        """Get information about the current model."""
        total_params = sum(p.numel() for p in self.network.parameters())
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'architecture': str(self.network)
        } 