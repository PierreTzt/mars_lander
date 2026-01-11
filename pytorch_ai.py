# -*- coding: utf-8 -*-
"""
Module IA avancee avec PyTorch
Deep Q-Network (DQN) avec:
- Reseau de neurones profond
- Experience Replay prioritaire
- Double DQN
- Dueling DQN architecture
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

# Detection GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[PyTorch] Device: {device}")


class DuelingDQN(nn.Module):
    """
    Architecture Dueling DQN
    Separe la valeur d'etat et l'avantage de chaque action
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super(DuelingDQN, self).__init__()

        # Couches communes (feature extraction)
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Branche Valeur d'etat V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

        # Branche Avantage A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )

    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class PrioritizedReplayBuffer:
    """
    Buffer de replay avec priorite
    Les experiences avec grande erreur TD sont echantillonnees plus souvent
    """

    def __init__(self, capacity: int = 100000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha  # Degre de prioritisation (0 = uniforme, 1 = full priorite)
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0

        experience = (state, action, reward, next_state, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        if len(self.buffer) == 0:
            return None

        # Calcul des probabilites
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # Echantillonnage
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)

        # Poids pour correction du biais
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        # Extraction des experiences
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            indices,
            weights.astype(np.float32)
        )

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6

    def __len__(self):
        return len(self.buffer)


class PyTorchDQNAgent:
    """
    Agent DQN complet avec PyTorch
    Features:
    - Double DQN (pour reduire la surestimation)
    - Dueling architecture
    - Prioritized Experience Replay
    - Soft target updates
    """

    def __init__(
        self,
        state_size: int = 8,
        n_angles: int = 37,  # -90 a +90 par pas de 5
        n_powers: int = 5,   # 0 a 4
        hidden_size: int = 256,
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.9995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        tau: float = 0.005,  # Soft update coefficient
        update_every: int = 4
    ):
        self.state_size = state_size
        self.n_angles = n_angles
        self.n_powers = n_powers
        self.action_size = n_angles * n_powers
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = tau
        self.update_every = update_every
        self.step_count = 0

        # Mapping actions
        self.actions = []
        for angle in range(-90, 91, 5):
            for power in range(5):
                self.actions.append((angle, power))

        # Reseaux
        self.policy_net = DuelingDQN(state_size, self.action_size, hidden_size).to(device)
        self.target_net = DuelingDQN(state_size, self.action_size, hidden_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimiseur avec gradient clipping
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Buffer de replay prioritaire
        self.memory = PrioritizedReplayBuffer(buffer_size)

        # Statistiques
        self.losses = []
        self.q_values_history = []
        self.episode_rewards = []
        self.current_episode_reward = 0

    def state_to_tensor(self, state) -> torch.Tensor:
        """Convertit l'etat en tensor normalise"""
        if isinstance(state, tuple):
            state = list(state)
        state_array = np.array(state, dtype=np.float32)
        # Normalisation
        # [x, y, h_speed, v_speed, fuel, angle, dist_zone, alt_zone]
        normalization = np.array([7000, 3000, 500, 500, 2000, 90, 7000, 3000], dtype=np.float32)
        state_array = state_array / normalization
        return torch.FloatTensor(state_array).unsqueeze(0).to(device)

    def choose_action(self, state) -> tuple:
        """
        Choisit une action avec politique epsilon-greedy
        Retourne (angle, puissance)
        """
        if random.random() < self.epsilon:
            # Exploration
            idx = random.randrange(self.action_size)
        else:
            # Exploitation
            with torch.no_grad():
                state_tensor = self.state_to_tensor(state)
                q_values = self.policy_net(state_tensor)
                idx = q_values.argmax().item()

                # Log Q-value max pour stats
                self.q_values_history.append(q_values.max().item())

        return self.actions[idx]

    def store_experience(self, state, action, reward, next_state, done):
        """Stocke une experience dans le buffer"""
        # Convertir action en index
        if action in self.actions:
            action_idx = self.actions.index(action)
        else:
            # Trouver l'action la plus proche
            action_idx = min(range(len(self.actions)),
                           key=lambda i: abs(self.actions[i][0] - action[0]) + abs(self.actions[i][1] - action[1]))

        state_array = np.array(state, dtype=np.float32)
        next_state_array = np.array(next_state, dtype=np.float32)

        self.memory.push(state_array, action_idx, reward, next_state_array, done)
        self.current_episode_reward += reward

        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0

    def train_step(self):
        """Effectue une etape d'entrainement"""
        if len(self.memory) < self.batch_size:
            return

        self.step_count += 1
        if self.step_count % self.update_every != 0:
            return

        # Echantillonnage prioritaire
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return

        states, actions, rewards, next_states, dones, indices, weights = batch

        # Conversion en tensors
        states = torch.FloatTensor(states / np.array([7000, 3000, 500, 500, 2000, 90, 7000, 3000])).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states / np.array([7000, 3000, 500, 500, 2000, 90, 7000, 3000])).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(device)

        # Calcul Q(s, a) actuel
        current_q = self.policy_net(states).gather(1, actions)

        # Double DQN: utilise policy_net pour selection, target_net pour evaluation
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Calcul erreur TD pour priorites
        td_errors = (current_q - target_q).detach().cpu().numpy().flatten()
        self.memory.update_priorities(indices, td_errors)

        # Loss ponderee par importance sampling
        loss = (weights * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        # Soft update du target network
        self._soft_update()

        self.losses.append(loss.item())

    def _soft_update(self):
        """Mise a jour douce du target network"""
        for target_param, policy_param in zip(self.target_net.parameters(),
                                               self.policy_net.parameters()):
            target_param.data.copy_(
                self.tau * policy_param.data + (1 - self.tau) * target_param.data
            )

    def decay_epsilon(self):
        """Decroit epsilon"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        """Sauvegarde le modele"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'losses': self.losses[-1000:],  # Derniers 1000
            'episode_rewards': self.episode_rewards[-1000:]
        }, filepath)
        print(f"[PyTorch] Modele sauvegarde: {filepath}")

    def load(self, filepath: str):
        """Charge le modele"""
        try:
            checkpoint = torch.load(filepath, map_location=device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.losses = checkpoint.get('losses', [])
            self.episode_rewards = checkpoint.get('episode_rewards', [])
            print(f"[PyTorch] Modele charge: {filepath}")
            return True
        except Exception as e:
            print(f"[PyTorch] Erreur chargement: {e}")
            return False

    def get_stats(self) -> dict:
        """Retourne les statistiques d'entrainement"""
        return {
            'epsilon': self.epsilon,
            'buffer_size': len(self.memory),
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,
            'avg_q': np.mean(self.q_values_history[-100:]) if self.q_values_history else 0,
            'avg_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'total_episodes': len(self.episode_rewards),
            'device': str(device)
        }


class ActorCritic(nn.Module):
    """
    Architecture Actor-Critic pour A2C/PPO
    Alternative au DQN pour des actions continues
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super(ActorCritic, self).__init__()

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Actor (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1)
        )

        # Critic (value)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        shared = self.shared(x)
        policy = self.actor(shared)
        value = self.critic(shared)
        return policy, value


class PPOAgent:
    """
    Agent PPO (Proximal Policy Optimization)
    Plus stable que DQN pour certains problemes
    """

    def __init__(
        self,
        state_size: int = 8,
        n_angles: int = 37,
        n_powers: int = 5,
        hidden_size: int = 256,
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        epsilon_clip: float = 0.2,
        k_epochs: int = 4
    ):
        self.state_size = state_size
        self.action_size = n_angles * n_powers
        self.gamma = gamma
        self.epsilon_clip = epsilon_clip
        self.k_epochs = k_epochs

        # Mapping actions
        self.actions = []
        for angle in range(-90, 91, 5):
            for power in range(5):
                self.actions.append((angle, power))

        self.policy = ActorCritic(state_size, self.action_size, hidden_size).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Buffer pour un episode
        self.states = []
        self.actions_taken = []
        self.rewards = []
        self.dones = []
        self.log_probs = []

    def choose_action(self, state) -> tuple:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            probs, _ = self.policy(state_tensor)

        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)

        self.log_probs.append(log_prob.item())

        return self.actions[action_idx.item()]

    def store_transition(self, state, action, reward, done):
        action_idx = self.actions.index(action) if action in self.actions else 0
        self.states.append(state)
        self.actions_taken.append(action_idx)
        self.rewards.append(reward)
        self.dones.append(done)

    def train(self):
        if len(self.states) == 0:
            return

        # Calcul des returns
        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)

        returns = torch.FloatTensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        states = torch.FloatTensor(self.states).to(device)
        actions = torch.LongTensor(self.actions_taken).to(device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(device)

        for _ in range(self.k_epochs):
            probs, values = self.policy(states)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)

            ratio = torch.exp(new_log_probs - old_log_probs)
            advantages = returns - values.squeeze().detach()

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values.squeeze(), returns)
            entropy_loss = -dist.entropy().mean()

            loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Clear buffer
        self.states.clear()
        self.actions_taken.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()


# Test du module
if __name__ == "__main__":
    print("=== Test PyTorch DQN Agent ===")
    agent = PyTorchDQNAgent()

    # Test avec etats aleatoires
    for i in range(100):
        state = np.random.rand(8) * [7000, 3000, 500, 500, 2000, 90, 7000, 3000]
        action = agent.choose_action(state)
        next_state = np.random.rand(8) * [7000, 3000, 500, 500, 2000, 90, 7000, 3000]
        reward = np.random.randn()
        done = np.random.random() < 0.1

        agent.store_experience(state, action, reward, next_state, done)
        agent.train_step()
        agent.decay_epsilon()

    print(f"Stats: {agent.get_stats()}")
    print("Test reussi!")
