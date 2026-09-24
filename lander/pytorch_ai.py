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
import os
import random

import numpy as np

# Normalisation des etats : [x, y, h_speed, v_speed, fuel, angle, dist_zone, alt_zone]
NORMALISATION_ETAT = np.array([7000, 3000, 500, 500, 2000, 90, 7000, 3000], dtype=np.float32)

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

    def __init__(self, capacity: int = 100000, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_increment: float = 1e-5):
        self.capacity = capacity
        self.alpha = alpha  # Degre de prioritisation (0 = uniforme, 1 = full priorite)
        # Correction du biais d'echantillonnage, augmentee progressivement jusqu'a 1
        self.beta = beta_start
        self.beta_increment = beta_increment
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

    def sample(self, batch_size: int):
        if len(self.buffer) == 0:
            return None

        beta = self.beta
        self.beta = min(1.0, self.beta + self.beta_increment)

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
        self.action_index = {action: i for i, action in enumerate(self.actions)}

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
        state_array = np.array(state, dtype=np.float32) / NORMALISATION_ETAT
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
        if action in self.action_index:
            action_idx = self.action_index[action]
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
        states = torch.FloatTensor(states / NORMALISATION_ETAT).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states / NORMALISATION_ETAT).to(device)
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
        """Decroit epsilon (a appeler une fois par episode)"""
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
        """Charge le modele s'il existe"""
        if not os.path.exists(filepath):
            return False
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
        if done:
            agent.decay_epsilon()

    print(f"Stats: {agent.get_stats()}")
    print("Test reussi!")
