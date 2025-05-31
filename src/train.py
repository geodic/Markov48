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
        
        # Reshape layer to treat the board as a 2D structure
        self.reshape = keras.layers.Reshape((4, 4, 1))
        
        # Convolutional layers for spatial pattern recognition
        self.conv1 = keras.layers.Conv2D(64, (2, 2), padding='same',
                                       kernel_initializer='he_uniform')
        self.bn1 = keras.layers.BatchNormalization()
        self.conv2 = keras.layers.Conv2D(128, (2, 2), padding='same',
                                       kernel_initializer='he_uniform')
        self.bn2 = keras.layers.BatchNormalization()
        
        # Process additional features
        self.aux_dense = keras.layers.Dense(64, activation='relu',
                                          kernel_initializer='he_uniform')
        self.aux_bn = keras.layers.BatchNormalization()
        
        # Combine spatial and auxiliary features
        self.flatten = keras.layers.Flatten()
        self.combine = keras.layers.Dense(256, activation='relu',
                                        kernel_initializer='he_uniform')
        self.bn3 = keras.layers.BatchNormalization()
        
        # Policy layers
        self.dense1 = keras.layers.Dense(128, activation='relu',
                                       kernel_initializer='he_uniform')
        self.bn4 = keras.layers.BatchNormalization()
        self.dropout = keras.layers.Dropout(0.2)
        self.output_layer = keras.layers.Dense(num_actions,
                                             kernel_initializer='glorot_uniform')
        
        # Value stream for better reward estimation
        self.value_dense = keras.layers.Dense(64, activation='relu',
                                            kernel_initializer='he_uniform')
        self.value_bn = keras.layers.BatchNormalization()
        self.value = keras.layers.Dense(1, kernel_initializer='glorot_uniform')
        
        # Advantage stream for better action selection
        self.advantage_dense = keras.layers.Dense(64, activation='relu',
                                                kernel_initializer='he_uniform')
        self.advantage_bn = keras.layers.BatchNormalization()
        self.advantage = keras.layers.Dense(num_actions,
                                          kernel_initializer='glorot_uniform')
    
    def call(self, x, training=False):
        # Split input into board state and auxiliary features
        board_values = tf.cast(x[:, :16], tf.float32)  # First 16 values are the board
        aux_features = tf.cast(x[:, 16:], tf.float32)  # Remaining features are auxiliary
        
        # Process board state through conv layers
        board = self.reshape(board_values)
        conv = tf.nn.relu(self.bn1(self.conv1(board), training=training))
        conv = tf.nn.relu(self.bn2(self.conv2(conv), training=training))
        conv_flat = self.flatten(conv)
        
        # Process auxiliary features
        aux = tf.nn.relu(self.aux_bn(self.aux_dense(aux_features), training=training))
        
        # Combine features
        combined = tf.concat([conv_flat, aux], axis=1)
        combined = tf.nn.relu(self.bn3(self.combine(combined), training=training))
        
        # Value stream
        value = tf.nn.relu(self.value_bn(self.value_dense(combined), training=training))
        value = self.value(value)
        
        # Advantage stream
        advantage = tf.nn.relu(self.advantage_bn(self.advantage_dense(combined), training=training))
        advantage = self.advantage(advantage)
        
        # Combine value and advantage (Dueling DQN architecture)
        q_values = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
        
        return q_values

    def build(self, input_shape):
        super().build(input_shape)  # This is important!

class Agent:
    def __init__(self, load_model=None):
        self.num_actions = 4
        self.memory = PriorityBuffer(maxlen=100000)  # Use PriorityBuffer instead of deque
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01  # Lower minimum exploration
        self.epsilon_decay = 0.9995  # Slower decay
        self.learning_rate = 0.001  # Start with slightly higher learning rate
        self.min_learning_rate = 0.0001
        self.lr_decay = 0.995
        self.model = DQN(self.num_actions)
        self.target_model = DQN(self.num_actions)
        self.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            clipnorm=1.0
        )
        self.update_target_counter = 0
        self.update_target_every = 10  # Less frequent updates for stability
        self.actions = ['up', 'down', 'left', 'right']
        self.training_steps = 0

        if load_model:
            self.load(load_model)
            self.epsilon = self.epsilon_min
    
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
        # Calculate TD error for prioritized replay
        state_tensor = tf.expand_dims(state, 0)
        next_state_tensor = tf.expand_dims(next_state, 0)
        
        current_q = self.model(state_tensor)[0].numpy()
        next_q = self.target_model(next_state_tensor)[0].numpy()
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(next_q)
        
        td_error = abs(target - current_q[action])
        self.memory.add((state, action, reward, next_state, done), td_error)
    
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and random.random() <= self.epsilon:
            return random.randrange(self.num_actions)
        
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_values = self.model(state_tensor)
        return tf.argmax(action_values[0]).numpy()
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return 0

        # Sample from prioritized replay buffer
        samples, indices, weights = self.memory.sample(batch_size)
        states = np.array([s[0] for s in samples])
        actions = np.array([s[1] for s in samples])
        rewards = np.array([s[2] for s in samples])
        next_states = np.array([s[3] for s in samples])
        dones = np.array([s[4] for s in samples])
        
        # Convert weights to tensor
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            # Current Q-values
            current_q = self.model(states, training=True)
            target_q = current_q.numpy()
            
            # Next Q-values from target network
            next_q = self.target_model(next_states, training=False)
            max_next_q = tf.reduce_max(next_q, axis=1)
            
            # Calculate targets
            for i in range(batch_size):
                if dones[i]:
                    target_q[i][actions[i]] = rewards[i]
                else:
                    target_q[i][actions[i]] = rewards[i] + self.gamma * max_next_q[i]
            
            # Calculate weighted loss
            losses = tf.keras.losses.MSE(target_q, current_q)
            total_loss = tf.reduce_mean(losses * weights)
        
        # Apply gradients
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        # Update priorities in memory
        td_errors = np.abs(target_q - current_q.numpy())
        self.memory.update_priorities(indices, td_errors[range(batch_size), actions])
        
        # Update exploration and learning rates
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.training_steps += 1
        if self.training_steps % 1000 == 0:  # Decay learning rate periodically
            current_lr = self.optimizer.learning_rate.numpy()
            if current_lr > self.min_learning_rate:
                new_lr = max(current_lr * self.lr_decay, self.min_learning_rate)
                self.optimizer.learning_rate.assign(new_lr)
        
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

class PriorityBuffer:
    def __init__(self, maxlen=100000, alpha=0.6, beta=0.4):
        self.memory = []
        self.maxlen = maxlen
        self.priorities = []
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.epsilon = 1e-6  # Small constant to prevent zero probabilities
    
    def add(self, experience, error=None):
        if error is None:
            priority = max(self.priorities) if self.priorities else 1.0
        else:
            priority = (abs(error) + self.epsilon) ** self.alpha
        
        if len(self.memory) >= self.maxlen:
            self.memory.pop(0)
            self.priorities.pop(0)
        
        self.memory.append(experience)
        self.priorities.append(priority)
    
    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            return [], [], []
        
        # Calculate sampling probabilities
        total = sum(self.priorities)
        probabilities = [p/total for p in self.priorities]
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = []
        max_weight = (min(probabilities) * len(self.memory)) ** (-self.beta)
        
        for idx in indices:
            prob = probabilities[idx]
            weight = (prob * len(self.memory)) ** (-self.beta)
            weights.append(weight / max_weight)
        
        samples = [self.memory[idx] for idx in indices]
        return samples, indices, weights
    
    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            if idx < len(self.priorities):
                self.priorities[idx] = (abs(error) + self.epsilon) ** self.alpha

    def __len__(self):
        return len(self.memory)

def calculate_reward(old_score, new_score, moved, highest_tile, merged_tiles, old_board, new_board):
    """Calculate reward based on multiple factors including board patterns and strategic positioning"""
    reward = 0
    
    # Base reward for score increase (logarithmic scaling with normalized factor)
    score_diff = new_score - old_score
    if score_diff > 0:
        reward += np.log2(score_diff + 1) / (np.log2(highest_tile + 1) + 1)  # Normalize by current game progress
    
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

def train_agent(episodes=1000, batch_size=64, save_dir='models'):  # Reduced batch size for better stability
    """Train the agent and save progress"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize agent
    agent = Agent()
    
    # Initialize metrics
    scores = []
    losses = []
    avg_scores = []
    best_score = 0
    no_improvement_count = 0
    best_avg_score = float('-inf')
    patience = 50  # Episodes to wait before early stopping
    
    # Training loop
    for episode in range(episodes):
        game = Board()
        state = agent.get_state(game.get_board())
        done = False
        episode_loss = 0
        moves = 0
        episode_rewards = []
        
        while not done:
            action_idx = agent.act(state)
            action = agent.actions[action_idx]
            
            old_score = game.get_score()
            old_board = game.get_board().copy()
            moved = game.move(action)
            new_score = game.get_score()
            new_board = game.get_board()
            
            highest_tile = np.max(new_board)
            merged_tiles = np.sum(new_board > old_board)
            reward = calculate_reward(old_score, new_score, moved, highest_tile, merged_tiles, old_board, new_board)
            episode_rewards.append(reward)
            
            next_state = agent.get_state(new_board)
            done = game.is_game_over()
            
            # Store experience and train
            agent.remember(state, action_idx, reward, next_state, done)
            loss = agent.replay(batch_size)
            if loss > 0:  # Only accumulate non-zero losses
                episode_loss += loss
            
            state = next_state
            moves += 1
            
            if moves > 1000:  # Prevent infinite games
                done = True
        
        # Update target network
        agent.update_target_counter += 1
        if agent.update_target_counter >= agent.update_target_every:
            agent.update_target_model()
            agent.update_target_counter = 0
        
        # Record metrics
        score = game.get_score()
        scores.append(score)
        avg_loss = episode_loss / moves if moves > 0 else 0
        losses.append(avg_loss)
        
        # Calculate average score
        if len(scores) > 100:
            avg_score = np.mean(scores[-100:])
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                no_improvement_count = 0
            else:
                no_improvement_count += 1
        else:
            avg_score = np.mean(scores)
        avg_scores.append(avg_score)
        
        # Save best model
        if score > best_score:
            best_score = score
            agent.save(f"{save_dir}/best_model_{timestamp}.weights.h5")
        
        # Print detailed progress
        if (episode + 1) % 1 == 0:
            print(f"\nEpisode: {episode+1}/{episodes}")
            print(f"Score: {score}, Best Score: {best_score}")
            print(f"Avg Score (100 ep): {avg_score:.2f}, Best Avg Score: {best_avg_score:.2f}")
            print(f"Epsilon: {agent.epsilon:.3f}, LR: {agent.optimizer.learning_rate.numpy():.6f}")
            print(f"Max Tile: {highest_tile}, Moves: {moves}")
            print(f"Avg Loss: {avg_loss:.4f}")
            print(f"Avg Reward: {np.mean(episode_rewards):.2f}")
            print("-" * 50)
            
            # Plot and save metrics
            plot_metrics(scores, avg_scores, losses, f"{save_dir}/training_progress_{timestamp}.png")
        
        # Early stopping check
        if no_improvement_count >= patience:
            print(f"\nStopping early - No improvement in average score for {patience} episodes")
            break
    
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
