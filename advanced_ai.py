"""
Systèmes d'IA avancés pour Mars Lander.

Ce module contient:
- Deep Q-Network (DQN) avec réseau de neurones
- Algorithme Génétique pour l'évolution de populations

Note: Ces implémentations utilisent numpy uniquement pour éviter
les dépendances lourdes comme PyTorch ou TensorFlow.
"""

import numpy as np
import random
import pickle
import os
from typing import List, Tuple, Dict, Optional, Any
from collections import deque
from dataclasses import dataclass, field


# =============================================================================
# DEEP Q-NETWORK (DQN) - Réseau de neurones simple
# =============================================================================

class NeuralNetwork:
    """
    Réseau de neurones simple implémenté avec numpy.
    Architecture: entrée -> couches cachées -> sortie
    """

    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.001):
        """
        Args:
            layer_sizes: Liste des tailles de couches [entrée, cachée1, ..., sortie]
            learning_rate: Taux d'apprentissage
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        # Initialisation Xavier/He
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    def relu(self, x: np.ndarray) -> np.ndarray:
        """Fonction d'activation ReLU."""
        return np.maximum(0, x)

    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Dérivée de ReLU."""
        return (x > 0).astype(float)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Propagation avant.

        Returns:
            (sortie, activations de chaque couche)
        """
        activations = [x]
        current = x

        for i in range(len(self.weights)):
            z = np.dot(current, self.weights[i]) + self.biases[i]
            # ReLU pour les couches cachées, linéaire pour la sortie
            if i < len(self.weights) - 1:
                current = self.relu(z)
            else:
                current = z  # Sortie linéaire pour les Q-values
            activations.append(current)

        return current, activations

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Prédit les Q-values pour un état."""
        output, _ = self.forward(x)
        return output

    def train_step(self, x: np.ndarray, target: np.ndarray) -> float:
        """
        Un pas d'entraînement avec backpropagation.

        Returns:
            Loss (MSE)
        """
        # Forward pass
        output, activations = self.forward(x)

        # Calcul de la loss
        loss = np.mean((output - target) ** 2)

        # Backward pass
        delta = 2 * (output - target) / output.shape[0]

        for i in range(len(self.weights) - 1, -1, -1):
            # Gradient des poids et biais
            dw = np.dot(activations[i].T, delta)
            db = np.sum(delta, axis=0, keepdims=True)

            # Mise à jour
            self.weights[i] -= self.learning_rate * dw
            self.biases[i] -= self.learning_rate * db

            # Propagation du gradient
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(activations[i])

        return loss

    def copy_weights_from(self, other: 'NeuralNetwork') -> None:
        """Copie les poids d'un autre réseau."""
        for i in range(len(self.weights)):
            self.weights[i] = other.weights[i].copy()
            self.biases[i] = other.biases[i].copy()

    def save(self, filepath: str) -> None:
        """Sauvegarde le réseau."""
        data = {
            'layer_sizes': self.layer_sizes,
            'learning_rate': self.learning_rate,
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases]
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filepath: str) -> bool:
        """Charge le réseau."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.layer_sizes = data['layer_sizes']
            self.learning_rate = data['learning_rate']
            self.weights = [np.array(w) for w in data['weights']]
            self.biases = [np.array(b) for b in data['biases']]
            return True
        return False


class DQNAgent:
    """
    Agent Deep Q-Network.
    Utilise un réseau de neurones pour approximer la fonction Q.
    """

    def __init__(self,
                 state_size: int = 8,
                 action_size: int = 65,
                 hidden_sizes: List[int] = [128, 64],
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 10000,
                 batch_size: int = 32,
                 target_update_freq: int = 100):
        """
        Args:
            state_size: Dimension de l'état
            action_size: Nombre d'actions possibles
            hidden_sizes: Tailles des couches cachées
            learning_rate: Taux d'apprentissage
            gamma: Facteur de discount
            epsilon: Taux d'exploration initial
            epsilon_min: Taux d'exploration minimum
            epsilon_decay: Décroissance de epsilon
            memory_size: Taille du replay buffer
            batch_size: Taille des batchs d'entraînement
            target_update_freq: Fréquence de mise à jour du réseau cible
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_step_count = 0

        # Réseaux principal et cible
        layer_sizes = [state_size] + hidden_sizes + [action_size]
        self.model = NeuralNetwork(layer_sizes, learning_rate)
        self.target_model = NeuralNetwork(layer_sizes, learning_rate)
        self.target_model.copy_weights_from(self.model)

        # Replay buffer
        self.memory = deque(maxlen=memory_size)

        # Statistiques
        self.losses: List[float] = []
        self.episodes_count = 0
        self.successful_landings = 0

        # Mapping des actions
        self.actions = self._generate_actions()

    def _generate_actions(self) -> List[Tuple[int, int]]:
        """Génère toutes les actions possibles (angle, puissance)."""
        actions = []
        for puissance in range(5):
            for angle_mult in range(-6, 7):
                angle = angle_mult * 15
                actions.append((angle, puissance))
        return actions

    def state_to_array(self, state: Tuple) -> np.ndarray:
        """Convertit un état tuple en array numpy normalisé."""
        # Normalisation des valeurs d'état
        x, y, vx, vy, angle, fuel, dist_zone, alt = state

        normalized = np.array([
            x / 7000,           # Position X normalisée
            y / 3000,           # Position Y normalisée
            vx / 100,           # Vitesse X normalisée
            vy / 100,           # Vitesse Y normalisée
            angle / 90,         # Angle normalisé
            fuel / 1000,        # Fuel normalisé
            dist_zone / 7000,   # Distance zone normalisée
            alt / 3000          # Altitude normalisée
        ]).reshape(1, -1)

        return normalized

    def choose_action(self, state: Tuple) -> Tuple[int, int]:
        """Choisit une action (epsilon-greedy)."""
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        state_array = self.state_to_array(state)
        q_values = self.model.predict(state_array)[0]
        action_idx = np.argmax(q_values)
        return self.actions[action_idx]

    def remember(self, state: Tuple, action: Tuple[int, int],
                 reward: float, next_state: Tuple, done: bool) -> None:
        """Stocke une expérience dans le replay buffer."""
        action_idx = self.actions.index(action) if action in self.actions else 0
        self.memory.append((state, action_idx, reward, next_state, done))

    def replay(self) -> float:
        """Entraîne sur un batch d'expériences."""
        if len(self.memory) < self.batch_size:
            return 0

        # Échantillonnage
        batch = random.sample(self.memory, self.batch_size)

        states = np.vstack([self.state_to_array(s) for s, _, _, _, _ in batch])
        next_states = np.vstack([self.state_to_array(ns) for _, _, _, ns, _ in batch])

        # Prédictions actuelles et cibles
        current_q = self.model.predict(states)
        target_q = self.target_model.predict(next_states)

        # Mise à jour des Q-values
        targets = current_q.copy()
        for i, (_, action_idx, reward, _, done) in enumerate(batch):
            if done:
                targets[i, action_idx] = reward
            else:
                targets[i, action_idx] = reward + self.gamma * np.max(target_q[i])

        # Entraînement
        loss = self.model.train_step(states, targets)
        self.losses.append(loss)

        # Mise à jour du réseau cible
        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.target_model.copy_weights_from(self.model)

        return loss

    def decay_epsilon(self) -> None:
        """Décroissance de epsilon."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str = "dqn_model.pkl") -> None:
        """Sauvegarde l'agent."""
        self.model.save(filepath)

    def load(self, filepath: str = "dqn_model.pkl") -> bool:
        """Charge l'agent."""
        if self.model.load(filepath):
            self.target_model.copy_weights_from(self.model)
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques."""
        return {
            'episodes': self.episodes_count,
            'successful_landings': self.successful_landings,
            'epsilon': self.epsilon,
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,
            'memory_size': len(self.memory)
        }


# =============================================================================
# ALGORITHME GÉNÉTIQUE
# =============================================================================

@dataclass
class Genome:
    """
    Un génome représentant un ensemble de règles de pilotage.
    Encode des poids pour un réseau de neurones simple.
    """
    id: int
    weights: np.ndarray
    fitness: float = 0.0
    age: int = 0
    landings: int = 0


class GeneticAlgorithm:
    """
    Algorithme génétique pour l'évolution de populations de pilotes.
    Chaque individu est un réseau de neurones avec des poids différents.
    """

    def __init__(self,
                 population_size: int = 50,
                 state_size: int = 8,
                 hidden_size: int = 32,
                 action_size: int = 65,
                 mutation_rate: float = 0.1,
                 mutation_strength: float = 0.3,
                 elite_ratio: float = 0.1,
                 crossover_rate: float = 0.7):
        """
        Args:
            population_size: Taille de la population
            state_size: Dimension de l'état
            hidden_size: Taille de la couche cachée
            action_size: Nombre d'actions
            mutation_rate: Probabilité de mutation
            mutation_strength: Force de la mutation
            elite_ratio: Ratio d'élites préservées
            crossover_rate: Probabilité de crossover
        """
        self.population_size = population_size
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.elite_count = int(population_size * elite_ratio)
        self.crossover_rate = crossover_rate

        # Calcul de la taille totale des poids
        self.weights_size = (state_size * hidden_size + hidden_size +
                            hidden_size * action_size + action_size)

        # Population
        self.population: List[Genome] = []
        self.generation = 0
        self.best_fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []

        # Actions possibles
        self.actions = self._generate_actions()

        # Initialisation
        self._initialize_population()

        # Individu actuel pour l'évaluation
        self.current_individual_idx = 0

    def _generate_actions(self) -> List[Tuple[int, int]]:
        """Génère les actions possibles."""
        actions = []
        for puissance in range(5):
            for angle_mult in range(-6, 7):
                angle = angle_mult * 15
                actions.append((angle, puissance))
        return actions

    def _initialize_population(self) -> None:
        """Initialise la population avec des génomes aléatoires."""
        self.population = []
        for i in range(self.population_size):
            weights = np.random.randn(self.weights_size) * 0.5
            self.population.append(Genome(id=i, weights=weights))

    def _weights_to_network(self, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                                np.ndarray, np.ndarray]:
        """Convertit les poids plats en matrices pour le réseau."""
        idx = 0

        # Couche d'entrée vers cachée
        w1_size = self.state_size * self.hidden_size
        w1 = weights[idx:idx + w1_size].reshape(self.state_size, self.hidden_size)
        idx += w1_size

        b1 = weights[idx:idx + self.hidden_size].reshape(1, self.hidden_size)
        idx += self.hidden_size

        # Couche cachée vers sortie
        w2_size = self.hidden_size * self.action_size
        w2 = weights[idx:idx + w2_size].reshape(self.hidden_size, self.action_size)
        idx += w2_size

        b2 = weights[idx:idx + self.action_size].reshape(1, self.action_size)

        return w1, b1, w2, b2

    def forward(self, genome: Genome, state: np.ndarray) -> np.ndarray:
        """Propagation avant pour un génome."""
        w1, b1, w2, b2 = self._weights_to_network(genome.weights)

        # Couche cachée (ReLU)
        hidden = np.maximum(0, np.dot(state, w1) + b1)

        # Sortie (linéaire)
        output = np.dot(hidden, w2) + b2

        return output

    def choose_action(self, state: Tuple) -> Tuple[int, int]:
        """Choisit une action pour l'individu actuel."""
        if self.current_individual_idx >= len(self.population):
            self.current_individual_idx = 0

        genome = self.population[self.current_individual_idx]

        # Normalisation de l'état
        x, y, vx, vy, angle, fuel, dist_zone, alt = state
        state_array = np.array([
            x / 7000, y / 3000, vx / 100, vy / 100,
            angle / 90, fuel / 1000, dist_zone / 7000, alt / 3000
        ]).reshape(1, -1)

        # Prédiction
        q_values = self.forward(genome, state_array)[0]
        action_idx = np.argmax(q_values)

        return self.actions[action_idx]

    def set_fitness(self, fitness: float, landed: bool = False) -> None:
        """Définit la fitness de l'individu actuel."""
        genome = self.population[self.current_individual_idx]
        genome.fitness = fitness
        if landed:
            genome.landings += 1

    def next_individual(self) -> bool:
        """
        Passe à l'individu suivant.
        Retourne True si on a évalué toute la population.
        """
        self.current_individual_idx += 1
        if self.current_individual_idx >= len(self.population):
            self._evolve()
            return True
        return False

    def _select_parent(self) -> Genome:
        """Sélection par tournoi."""
        tournament_size = 5
        candidates = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(candidates, key=lambda g: g.fitness)

    def _crossover(self, parent1: Genome, parent2: Genome) -> np.ndarray:
        """Crossover uniforme."""
        mask = np.random.random(self.weights_size) < 0.5
        child_weights = np.where(mask, parent1.weights, parent2.weights)
        return child_weights

    def _mutate(self, weights: np.ndarray) -> np.ndarray:
        """Mutation gaussienne."""
        mutation_mask = np.random.random(self.weights_size) < self.mutation_rate
        mutations = np.random.randn(self.weights_size) * self.mutation_strength
        weights = weights + mutation_mask * mutations
        return weights

    def _evolve(self) -> None:
        """Fait évoluer la population."""
        # Tri par fitness
        self.population.sort(key=lambda g: g.fitness, reverse=True)

        # Statistiques
        best_fitness = self.population[0].fitness
        avg_fitness = np.mean([g.fitness for g in self.population])
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)

        # Nouvelle population
        new_population = []

        # Élitisme: garder les meilleurs
        for i in range(self.elite_count):
            elite = Genome(
                id=i,
                weights=self.population[i].weights.copy(),
                fitness=0,
                age=self.population[i].age + 1,
                landings=self.population[i].landings
            )
            new_population.append(elite)

        # Reproduction
        while len(new_population) < self.population_size:
            parent1 = self._select_parent()
            parent2 = self._select_parent()

            if random.random() < self.crossover_rate:
                child_weights = self._crossover(parent1, parent2)
            else:
                child_weights = parent1.weights.copy()

            child_weights = self._mutate(child_weights)

            new_population.append(Genome(
                id=len(new_population),
                weights=child_weights
            ))

        self.population = new_population
        self.generation += 1
        self.current_individual_idx = 0

        print(f"Generation {self.generation} | Best: {best_fitness:.2f} | Avg: {avg_fitness:.2f}")

    def get_best_genome(self) -> Genome:
        """Retourne le meilleur génome."""
        return max(self.population, key=lambda g: g.fitness)

    def save(self, filepath: str = "genetic_population.pkl") -> None:
        """Sauvegarde la population."""
        data = {
            'population': [(g.id, g.weights.tolist(), g.fitness, g.age, g.landings)
                          for g in self.population],
            'generation': self.generation,
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filepath: str = "genetic_population.pkl") -> bool:
        """Charge la population."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.population = [
                Genome(id=g[0], weights=np.array(g[1]), fitness=g[2], age=g[3], landings=g[4])
                for g in data['population']
            ]
            self.generation = data['generation']
            self.best_fitness_history = data['best_fitness_history']
            self.avg_fitness_history = data['avg_fitness_history']
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques."""
        best = self.get_best_genome()
        return {
            'generation': self.generation,
            'population_size': len(self.population),
            'best_fitness': best.fitness,
            'best_landings': best.landings,
            'avg_fitness': np.mean([g.fitness for g in self.population]),
            'current_individual': self.current_individual_idx
        }


# =============================================================================
# PLANÈTES AVEC GRAVITÉS DIFFÉRENTES
# =============================================================================

@dataclass
class Planet:
    """Représente une planète avec ses caractéristiques."""
    name: str
    gravity: float              # m/s²
    atmosphere_density: float   # 0-1 (0 = vide, 1 = dense)
    surface_color: Tuple[int, int, int]
    sky_color: Tuple[int, int, int]
    wind_factor: float          # Multiplicateur de vent
    description: str


class PlanetSystem:
    """Gère les différentes planètes disponibles."""

    def __init__(self):
        self.planets: Dict[str, Planet] = {
            'mars': Planet(
                name="Mars",
                gravity=3.72,
                atmosphere_density=0.01,
                surface_color=(193, 68, 14),
                sky_color=(20, 10, 5),
                wind_factor=0.5,
                description="La planète rouge - Gravité faible, atmosphère ténue"
            ),
            'moon': Planet(
                name="Lune",
                gravity=1.62,
                atmosphere_density=0.0,
                surface_color=(128, 128, 128),
                sky_color=(0, 0, 0),
                wind_factor=0.0,
                description="Notre satellite - Pas d'atmosphère, gravité très faible"
            ),
            'earth': Planet(
                name="Terre",
                gravity=9.81,
                atmosphere_density=1.0,
                surface_color=(34, 139, 34),
                sky_color=(135, 206, 235),
                wind_factor=1.0,
                description="Notre planète - Gravité standard, atmosphère dense"
            ),
            'europa': Planet(
                name="Europa",
                gravity=1.31,
                atmosphere_density=0.0,
                surface_color=(200, 220, 255),
                sky_color=(5, 5, 20),
                wind_factor=0.0,
                description="Lune de Jupiter - Surface glacée, très faible gravité"
            ),
            'titan': Planet(
                name="Titan",
                gravity=1.35,
                atmosphere_density=1.5,
                surface_color=(180, 140, 60),
                sky_color=(255, 180, 100),
                wind_factor=2.0,
                description="Lune de Saturne - Atmosphère très dense, vents forts"
            ),
            'venus': Planet(
                name="Vénus",
                gravity=8.87,
                atmosphere_density=90.0,
                surface_color=(200, 150, 50),
                sky_color=(255, 200, 100),
                wind_factor=3.0,
                description="L'enfer - Gravité forte, pression atmosphérique écrasante"
            ),
        }

        self.current_planet = 'mars'

    def get_current_planet(self) -> Planet:
        """Retourne la planète actuelle."""
        return self.planets[self.current_planet]

    def set_planet(self, name: str) -> bool:
        """Change de planète."""
        if name.lower() in self.planets:
            self.current_planet = name.lower()
            return True
        return False

    def get_gravity_factor(self) -> float:
        """Retourne le facteur de gravité relatif à Mars."""
        mars_gravity = 3.72
        return self.planets[self.current_planet].gravity / mars_gravity

    def get_wind_factor(self) -> float:
        """Retourne le facteur de vent."""
        return self.planets[self.current_planet].wind_factor

    def get_planet_list(self) -> List[str]:
        """Retourne la liste des planètes."""
        return list(self.planets.keys())

    def draw_info(self, screen, font, x: int = 10, y: int = 10) -> None:
        """Affiche les informations de la planète."""
        planet = self.get_current_planet()

        # Nom
        name_text = font.render(f"Planet: {planet.name}", True, planet.surface_color)
        screen.blit(name_text, (x, y))

        # Gravité
        grav_text = font.render(f"Gravity: {planet.gravity} m/s²", True, (200, 200, 200))
        screen.blit(grav_text, (x, y + 20))
