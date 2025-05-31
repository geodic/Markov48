import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Enable oneDNN optimizations
os.environ['TF_NUM_INTEROP_THREADS'] = '1'  # Number of threads used to parallelize independent operations
os.environ['TF_NUM_INTRAOP_THREADS'] = '4'  # Number of threads used to parallelize operations
os.environ['OMP_NUM_THREADS'] = '4'  # OpenMP threads for parallel processing
os.environ['KMP_BLOCKTIME'] = '0'  # Time in ms a thread should wait before sleeping
os.environ['KMP_AFFINITY'] = 'granularity=fine,verbose,compact,1,0'  # Thread affinity control

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Enable memory growth to avoid allocating all GPU memory at once
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    try:
        tf.config.experimental.set_memory_growth(device, True)
    except:
        pass

# Configure TensorFlow to use Intel optimizations
tf.config.optimizer.set_jit(True)  # Enable XLA JIT compilation
tf.config.threading.set_intra_op_parallelism_threads(4)  # Use 4 threads for parallelizing operations
tf.config.threading.set_inter_op_parallelism_threads(1)  # Use 1 thread for independent operations
from collections import deque
import random
from src.board import Board
import matplotlib.pyplot as plt
from datetime import datetime

# 1. Define the Deep Q-Network architecture
class DQN(keras.Model):
    def __init__(self, num_actions):
        super().__init__()
        # Neural network layers with increased width for enhanced state space
        self.input_layer = keras.layers.Input(shape=(40,))
        self.dense1 = keras.layers.Dense(512, activation='relu')
        self.dense2 = keras.layers.Dense(256, activation='relu')
        self.dense3 = keras.layers.Dense(128, activation='relu')
        self.dense4 = keras.layers.Dense(num_actions)  # Output layer (4 actions: up, down, left, right)
    
    def call(self, x):
        x = tf.cast(x, tf.float32)  # Convert input to float32
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return x

    def build(self, input_shape):
        super().build(input_shape)  # This is important!

class Agent:
    def __init__(self, load_model=None):
        self.num_actions = 4  # up, down, left, right
        self.memory = deque(maxlen=100000)
        self.gamma = 0.95    # discount factor
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(self.num_actions)
        self.target_model = DQN(self.num_actions)  # Target network for stable training
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.update_target_counter = 0
        self.update_target_every = 10  # Update target network every N episodes
        
        # Direction mapping
        self.actions = ['up', 'down', 'left', 'right']

        if load_model:
            self.load(load_model)
            self.epsilon = self.epsilon_min  # Start with minimal exploration if loading model
    
    def get_state(self, board):
        """Convert board state to a feature vector containing:
        1. Normalized cell values
        2. Mergeable pairs count
        3. Empty cells binary mask
        4. Pattern-based features (monotonicity and smoothness)
        """
        board_array = np.array(board)
        features = []
        
        # 1. Normalized cell values
        normalized_values = np.log2(board_array + 1)
        features.extend(normalized_values.flatten())
        
        # 2. Count mergeable pairs
        mergeable_h = sum(np.sum(board_array[:, i] == board_array[:, i+1]) for i in range(board_array.shape[1]-1))
        mergeable_v = sum(np.sum(board_array[i, :] == board_array[i+1, :]) for i in range(board_array.shape[0]-1))
        features.extend([float(mergeable_h), float(mergeable_v)])
        
        # 3. Empty cells binary mask (1 for empty, 0 for filled)
        empty_mask = (board_array == 0).astype(float)
        features.extend(empty_mask.flatten())
        
        # 4. Pattern-based features
        # Monotonicity: penalty for non-monotonic sequences
        def get_monotonicity(arr):
            return -sum(abs(np.diff(np.log2(arr + 1))))
        
        # Calculate monotonicity in all directions
        monotonicity_lr = sum(get_monotonicity(row) for row in board_array)
        monotonicity_rl = sum(get_monotonicity(row[::-1]) for row in board_array)
        monotonicity_ud = sum(get_monotonicity(col) for col in board_array.T)
        monotonicity_du = sum(get_monotonicity(col[::-1]) for col in board_array.T)
        
        # Smoothness: similarity between adjacent tiles
        def get_smoothness(arr):
            return -sum(abs(np.log2(arr[:-1] + 1) - np.log2(arr[1:] + 1)))
        
        smoothness_h = sum(get_smoothness(row) for row in board_array)
        smoothness_v = sum(get_smoothness(col) for col in board_array.T)
        
        features.extend([
            monotonicity_lr, monotonicity_rl,
            monotonicity_ud, monotonicity_du,
            smoothness_h, smoothness_v
        ])
        
        return np.array(features)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and random.random() <= self.epsilon:
            return random.randrange(self.num_actions)
        
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_values = self.model(state_tensor)
        return tf.argmax(action_values[0]).numpy()
    
    def replay(self, batch_size):
        """Train on a batch of experiences"""
        if len(self.memory) < batch_size:
            return 0
        
        minibatch = random.sample(self.memory, batch_size)
        total_loss = 0
        
        states = np.array([x[0] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])
        
        # Predict Q-values for current and next states using target network
        current_q_values = self.model(states)
        next_q_values = self.target_model(next_states)
        
        X = []
        Y = []
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            if done:
                target = reward
            else:
                next_q = next_q_values[i].numpy()
                target = reward + self.gamma * np.max(next_q)
            
            target_q_values = current_q_values[i].numpy()
            target_q_values[action] = target
            
            X.append(state)
            Y.append(target_q_values)
        
        X = np.array(X)
        Y = np.array(Y)
        
        # Create loss function instance outside the tape for efficiency
        mse = tf.keras.losses.MeanSquaredError()
        
        with tf.GradientTape() as tape:
            q_values = self.model(X)
            total_loss = mse(Y, q_values)
        
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return float(total_loss)

    def update_target_model(self):
        """Update target model with weights from main model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def save(self, filepath):
        """Save model weights"""
        self.model.save_weights(filepath)
    
    def load(self, filepath):
        """Load model weights"""
        self.model.load_weights(filepath)
        self.target_model.set_weights(self.model.get_weights())

def calculate_reward(old_score, new_score, moved, highest_tile, merged_tiles, old_board, new_board):
    """Calculate reward based on multiple factors including board patterns and strategic positioning"""
    reward = 0
    
    # Base reward for score increase (logarithmic scaling to prevent exploitation)
    score_diff = new_score - old_score
    if score_diff > 0:
        reward += np.log2(score_diff + 1) * 2
    
    # Strategic rewards
    if moved:
        # Analyze board state after move
        board = np.array(new_board)
        size = len(board)
        
        # 1. Corner placement reward
        corners = [board[0,0], board[0,size-1], board[size-1,0], board[size-1,size-1]]
        max_corner = max(corners)
        if max_corner == highest_tile and highest_tile >= 64:
            reward += np.log2(highest_tile) * 0.5
        
        # 2. Monotonicity reward (encouraging ordered sequences)
        def monotonicity_reward(arr):
            diffs = np.diff(np.log2(arr + 1))
            return np.sum(diffs[diffs <= 0]) * 0.1  # Reward for descending order
        
        # Check monotonicity in all directions
        for row in board:
            reward += monotonicity_reward(row)  # Left to right
            reward += monotonicity_reward(row[::-1])  # Right to left
        for col in board.T:
            reward += monotonicity_reward(col)  # Top to bottom
            reward += monotonicity_reward(col[::-1])  # Bottom to top
        
        # 3. Merging rewards (weighted by tile value)
        if merged_tiles > 0:
            # Higher reward for merging larger tiles
            merged_value = np.sum(board[board > old_board])
            reward += np.log2(merged_value + 1) * 0.3
        
        # 4. Empty cell rewards (exponential scaling)
        empty_cells = np.sum(board == 0)
        reward += (1.5 ** empty_cells - 1) * 0.1
        
        # 5. Adjacency penalty (encourage similar values to be adjacent)
        def adjacency_penalty():
            penalty = 0
            for i in range(size):
                for j in range(size):
                    if board[i,j] != 0:
                        current = np.log2(board[i,j] + 1)
                        # Check horizontal neighbor
                        if j < size-1 and board[i,j+1] != 0:
                            neighbor = np.log2(board[i,j+1] + 1)
                            penalty += abs(current - neighbor) * 0.1
                        # Check vertical neighbor
                        if i < size-1 and board[i+1,j] != 0:
                            neighbor = np.log2(board[i+1,j] + 1)
                            penalty += abs(current - neighbor) * 0.1
            return penalty
        
        reward -= adjacency_penalty()
    else:
        # Increased penalty for invalid moves
        reward -= 20  # More severe than before to discourage random moves
    
    # Progressive rewards for achieving higher tiles
    tile_rewards = {
        2048: 2000,
        1024: 1000,
        512: 500,
        256: 250,
        128: 100,
        64: 50
    }
    for tile, bonus in tile_rewards.items():
        if highest_tile >= tile:
            reward += bonus
            break  # Only give the highest applicable bonus
    
    # Small move penalty scaled by board state
    move_penalty = 0.2 * (1 + np.log2(highest_tile + 1) / 11)  # Increases as the game progresses
    reward -= move_penalty
    
    return reward

def plot_metrics(scores, avg_scores, losses, filename):
    """Plot and save training metrics"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot scores
    ax1.plot(scores, label='Score', alpha=0.6)
    ax1.plot(avg_scores, label='Average Score (100 episodes)', linewidth=2)
    ax1.set_title('Training Scores')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.legend()
    ax1.grid(True)
    
    # Plot losses
    ax2.plot(losses, label='Loss', alpha=0.6)
    ax2.set_title('Training Loss')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def evaluate_agent(agent, num_games=100):
    """Evaluate agent performance without training"""
    scores = []
    max_tiles = []
    
    for _ in range(num_games):
        game = Board()
        done = False
        while not done:
            state = agent.get_state(game.get_board())
            action_idx = agent.act(state, training=False)
            action = agent.actions[action_idx]
            moved = game.move(action)
            done = game.is_game_over()
        
        scores.append(game.get_score())
        max_tiles.append(np.max(game.get_board()))
    
    return {
        'avg_score': np.mean(scores),
        'max_score': np.max(scores),
        'avg_max_tile': np.mean(max_tiles),
        'max_tile': np.max(max_tiles)
    }

def train_agent(episodes=1000, batch_size=64, save_dir='models'):
    """Train the agent and save progress"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Configure training optimizations
    agent = Agent()
    agent.learning_rate = 0.0005  # Slightly lower learning rate for stability
    agent.optimizer = keras.optimizers.Adam(
        learning_rate=agent.learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    
    scores = []
    losses = []
    avg_scores = []
    best_score = 0
    
    for episode in range(episodes):
        game = Board()
        state = agent.get_state(game.get_board())
        done = False
        episode_loss = 0
        moves = 0
        
        while not done:
            # Choose and perform action
            action_idx = agent.act(state)
            action = agent.actions[action_idx]
            
            # Get reward and next state
            old_score = game.get_score()
            old_board = game.get_board().copy()
            moved = game.move(action)
            new_score = game.get_score()
            new_board = game.get_board()
            
            # Calculate reward
            highest_tile = np.max(new_board)
            merged_tiles = np.sum(new_board > old_board)
            reward = calculate_reward(old_score, new_score, moved, highest_tile, merged_tiles, old_board, new_board)
            
            next_state = agent.get_state(new_board)
            done = game.is_game_over()
            
            # Store experience and train
            agent.remember(state, action_idx, reward, next_state, done)
            loss = agent.replay(batch_size)
            episode_loss += loss
            
            state = next_state
            moves += 1
            
            if moves > 1000:  # Prevent infinite games
                done = True
        
        # Update target network periodically
        agent.update_target_counter += 1
        if agent.update_target_counter >= agent.update_target_every:
            agent.update_target_model()
            agent.update_target_counter = 0
        
        # Record metrics
        score = game.get_score()
        scores.append(score)
        losses.append(episode_loss / moves if moves > 0 else 0)
        
        # Calculate average score
        if len(scores) > 100:
            avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)
        else:
            avg_score = np.mean(scores)
            avg_scores.append(avg_score)
        
        # Save best model
        if score > best_score:
            best_score = score
            agent.save(f"{save_dir}/best_model_{timestamp}.weights.h5")
        
        # Print progress
        if (episode + 1) % 1 == 0:
            print(f"Episode: {episode+1}/{episodes}")
            print(f"Score: {score}, Avg Score: {avg_score:.2f}")
            print(f"Epsilon: {agent.epsilon:.3f}, Max Tile: {np.max(game.get_board())}")
            print(f"Avg Loss: {episode_loss/moves if moves > 0 else 0:.4f}")
            print("-" * 50)
            
            # Plot and save metrics
            plot_metrics(scores, avg_scores, losses, f"{save_dir}/training_progress_{timestamp}.png")
    
    # Final evaluation
    print("\nTraining completed. Running evaluation...")
    eval_results = evaluate_agent(agent)
    print("\nEvaluation Results:")
    print(f"Average Score: {eval_results['avg_score']:.2f}")
    print(f"Max Score: {eval_results['max_score']}")
    print(f"Average Max Tile: {eval_results['avg_max_tile']}")
    print(f"Highest Tile Achieved: {eval_results['max_tile']}")
    
    return agent, scores, losses

if __name__ == "__main__":
    train_agent(episodes=1000)
