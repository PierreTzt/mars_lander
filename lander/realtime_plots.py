# -*- coding: utf-8 -*-
"""
Module de visualisation temps reel avec Matplotlib
Graphiques de l'apprentissage de l'IA:
- Courbe de recompenses
- Taux de succes
- Evolution epsilon
- Heatmaps d'etats visites
- Distribution des actions
"""

import matplotlib
matplotlib.use('TkAgg')  # Backend pour affichage interactif

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import numpy as np
from collections import deque
from typing import TYPE_CHECKING
import threading
import queue

if TYPE_CHECKING:
    import pygame
import time


class RealtimeLearningPlots:
    """
    Fenetre de graphiques temps reel pour visualiser l'apprentissage
    """

    def __init__(self, max_history: int = 1000, update_interval: int = 100):
        self.max_history = max_history
        self.update_interval = update_interval

        # Historiques des donnees
        self.rewards_history = deque(maxlen=max_history)
        self.success_history = deque(maxlen=max_history)
        self.epsilon_history = deque(maxlen=max_history)
        self.loss_history = deque(maxlen=max_history)
        self.q_values_history = deque(maxlen=max_history)

        # Heatmap des etats
        self.state_heatmap = np.zeros((50, 70))  # y, x grille

        # Distribution des actions
        self.action_counts = np.zeros((37, 5))  # angles x puissances

        # Statistiques
        self.total_episodes = 0
        self.total_landings = 0
        self.total_crashes = 0

        # Queue pour communication thread-safe
        self.data_queue = queue.Queue()

        # Thread et fenetre
        self.plot_thread = None
        self.running = False
        self.fig = None
        self.axes = {}

    def start(self):
        """Demarre la fenetre de plots dans un thread separe"""
        if self.running:
            return

        self.running = True
        self.plot_thread = threading.Thread(target=self._run_plot_window, daemon=True)
        self.plot_thread.start()

    def stop(self):
        """Arrete la fenetre de plots"""
        self.running = False
        if self.plot_thread:
            self.plot_thread.join(timeout=1)

    def _run_plot_window(self):
        """Execute la fenetre matplotlib dans son thread"""
        # Creation de la figure
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.suptitle('Mars Lander - Apprentissage IA en Temps Reel', fontsize=14, fontweight='bold')

        # Grille de subplots
        gs = GridSpec(3, 3, figure=self.fig, hspace=0.35, wspace=0.3)

        # Subplot 1: Recompenses
        self.axes['rewards'] = self.fig.add_subplot(gs[0, 0])
        self.axes['rewards'].set_title('Recompenses par Episode')
        self.axes['rewards'].set_xlabel('Episode')
        self.axes['rewards'].set_ylabel('Recompense')
        self.lines_rewards, = self.axes['rewards'].plot([], [], 'b-', alpha=0.5, label='Recompense')
        self.lines_rewards_avg, = self.axes['rewards'].plot([], [], 'r-', linewidth=2, label='Moyenne (50)')
        self.axes['rewards'].legend(loc='upper left')
        self.axes['rewards'].grid(True, alpha=0.3)

        # Subplot 2: Taux de succes
        self.axes['success'] = self.fig.add_subplot(gs[0, 1])
        self.axes['success'].set_title('Taux de Succes')
        self.axes['success'].set_xlabel('Episode')
        self.axes['success'].set_ylabel('Taux (%)')
        self.lines_success, = self.axes['success'].plot([], [], 'g-', linewidth=2)
        self.axes['success'].set_ylim(0, 100)
        self.axes['success'].grid(True, alpha=0.3)
        self.axes['success'].axhline(y=80, color='r', linestyle='--', alpha=0.5, label='Objectif 80%')

        # Subplot 3: Epsilon
        self.axes['epsilon'] = self.fig.add_subplot(gs[0, 2])
        self.axes['epsilon'].set_title('Epsilon (Exploration)')
        self.axes['epsilon'].set_xlabel('Episode')
        self.axes['epsilon'].set_ylabel('Epsilon')
        self.lines_epsilon, = self.axes['epsilon'].plot([], [], 'm-', linewidth=2)
        self.axes['epsilon'].set_ylim(0, 1)
        self.axes['epsilon'].grid(True, alpha=0.3)

        # Subplot 4: Loss
        self.axes['loss'] = self.fig.add_subplot(gs[1, 0])
        self.axes['loss'].set_title('Loss (Erreur)')
        self.axes['loss'].set_xlabel('Step')
        self.axes['loss'].set_ylabel('Loss')
        self.lines_loss, = self.axes['loss'].plot([], [], 'orange', alpha=0.5)
        self.lines_loss_avg, = self.axes['loss'].plot([], [], 'red', linewidth=2)
        self.axes['loss'].grid(True, alpha=0.3)

        # Subplot 5: Q-Values
        self.axes['qvalues'] = self.fig.add_subplot(gs[1, 1])
        self.axes['qvalues'].set_title('Q-Values Moyennes')
        self.axes['qvalues'].set_xlabel('Step')
        self.axes['qvalues'].set_ylabel('Q-Value')
        self.lines_qvalues, = self.axes['qvalues'].plot([], [], 'c-', linewidth=2)
        self.axes['qvalues'].grid(True, alpha=0.3)

        # Subplot 6: Statistiques texte
        self.axes['stats'] = self.fig.add_subplot(gs[1, 2])
        self.axes['stats'].axis('off')
        self.stats_text = self.axes['stats'].text(
            0.1, 0.9, '', transform=self.axes['stats'].transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        # Subplot 7: Heatmap des positions
        self.axes['heatmap'] = self.fig.add_subplot(gs[2, 0:2])
        self.axes['heatmap'].set_title('Heatmap des Positions Visitees')
        self.axes['heatmap'].set_xlabel('Position X')
        self.axes['heatmap'].set_ylabel('Position Y')
        self.heatmap_img = self.axes['heatmap'].imshow(
            self.state_heatmap, aspect='auto', cmap='hot',
            extent=[0, 7000, 0, 3000], origin='lower'
        )
        plt.colorbar(self.heatmap_img, ax=self.axes['heatmap'], label='Visites')

        # Subplot 8: Distribution des actions
        self.axes['actions'] = self.fig.add_subplot(gs[2, 2])
        self.axes['actions'].set_title('Distribution des Actions')
        self.axes['actions'].set_xlabel('Puissance')
        self.axes['actions'].set_ylabel('Angle')
        self.action_img = self.axes['actions'].imshow(
            self.action_counts, aspect='auto', cmap='viridis',
            extent=[0, 5, -90, 90], origin='lower'
        )

        plt.tight_layout()

        # Animation
        # Garder une reference, sinon l'animation est detruite par le ramasse-miettes
        self._animation = FuncAnimation(
            self.fig, self._update_plots,
            interval=self.update_interval,
            blit=False, cache_frame_data=False
        )

        plt.show()

    def _update_plots(self, frame):
        """Met a jour tous les graphiques"""
        # Traiter les donnees en queue
        while not self.data_queue.empty():
            try:
                data_type, data = self.data_queue.get_nowait()
                self._process_data(data_type, data)
            except queue.Empty:
                break

        # Mise a jour des courbes
        if self.rewards_history:
            x = list(range(len(self.rewards_history)))
            self.lines_rewards.set_data(x, list(self.rewards_history))

            # Moyenne mobile
            if len(self.rewards_history) >= 50:
                avg = np.convolve(list(self.rewards_history), np.ones(50)/50, mode='valid')
                self.lines_rewards_avg.set_data(range(49, len(self.rewards_history)), avg)

            self.axes['rewards'].relim()
            self.axes['rewards'].autoscale_view()

        if self.success_history:
            x = list(range(len(self.success_history)))
            self.lines_success.set_data(x, list(self.success_history))
            self.axes['success'].relim()
            self.axes['success'].autoscale_view()

        if self.epsilon_history:
            x = list(range(len(self.epsilon_history)))
            self.lines_epsilon.set_data(x, list(self.epsilon_history))
            self.axes['epsilon'].relim()
            self.axes['epsilon'].autoscale_view()

        if self.loss_history:
            x = list(range(len(self.loss_history)))
            self.lines_loss.set_data(x, list(self.loss_history))

            if len(self.loss_history) >= 50:
                avg = np.convolve(list(self.loss_history), np.ones(50)/50, mode='valid')
                self.lines_loss_avg.set_data(range(49, len(self.loss_history)), avg)

            self.axes['loss'].relim()
            self.axes['loss'].autoscale_view()

        if self.q_values_history:
            x = list(range(len(self.q_values_history)))
            self.lines_qvalues.set_data(x, list(self.q_values_history))
            self.axes['qvalues'].relim()
            self.axes['qvalues'].autoscale_view()

        # Mise a jour statistiques
        success_rate = (self.total_landings / max(1, self.total_episodes)) * 100
        stats_str = (
            f"Episodes: {self.total_episodes}\n"
            f"Atterrissages: {self.total_landings}\n"
            f"Crashes: {self.total_crashes}\n"
            f"Taux succes: {success_rate:.1f}%\n"
            f"Epsilon: {self.epsilon_history[-1] if self.epsilon_history else 0:.4f}\n"
            f"Derniere recompense: {self.rewards_history[-1] if self.rewards_history else 0:.2f}"
        )
        self.stats_text.set_text(stats_str)

        # Mise a jour heatmap
        self.heatmap_img.set_data(self.state_heatmap)
        self.heatmap_img.set_clim(vmin=0, vmax=max(1, self.state_heatmap.max()))

        # Mise a jour actions
        self.action_img.set_data(self.action_counts)
        self.action_img.set_clim(vmin=0, vmax=max(1, self.action_counts.max()))

        return []

    def _process_data(self, data_type: str, data):
        """Traite les donnees recues"""
        if data_type == 'episode':
            reward, success, epsilon = data
            self.rewards_history.append(reward)
            self.epsilon_history.append(epsilon)
            self.total_episodes += 1

            if success:
                self.total_landings += 1
            else:
                self.total_crashes += 1

            # Calcul taux de succes mobile
            success_rate = (self.total_landings / max(1, self.total_episodes)) * 100
            self.success_history.append(success_rate)

        elif data_type == 'loss':
            self.loss_history.append(data)

        elif data_type == 'qvalue':
            self.q_values_history.append(data)

        elif data_type == 'state':
            x, y = data
            # Convertir en indices de grille
            ix = min(69, max(0, int(x / 100)))
            iy = min(49, max(0, int(y / 60)))
            self.state_heatmap[iy, ix] += 1

        elif data_type == 'action':
            angle, power = data
            # Convertir angle en index
            angle_idx = (angle + 90) // 5
            angle_idx = min(36, max(0, angle_idx))
            power_idx = min(4, max(0, power))
            self.action_counts[angle_idx, power_idx] += 1

    def log_episode(self, reward: float, success: bool, epsilon: float):
        """Enregistre les resultats d'un episode"""
        self.data_queue.put(('episode', (reward, success, epsilon)))

    def log_loss(self, loss: float):
        """Enregistre une valeur de loss"""
        self.data_queue.put(('loss', loss))

    def log_qvalue(self, qvalue: float):
        """Enregistre une Q-value"""
        self.data_queue.put(('qvalue', qvalue))

    def log_state(self, x: float, y: float):
        """Enregistre une position visitee"""
        self.data_queue.put(('state', (x, y)))

    def log_action(self, angle: int, power: int):
        """Enregistre une action"""
        self.data_queue.put(('action', (angle, power)))


class EmbeddedPlots:
    """
    Version des plots embeddable dans pygame
    Genere des surfaces pygame a partir de matplotlib
    """

    def __init__(self, width: int = 400, height: int = 300):
        self.width = width
        self.height = height

        # Historiques
        self.rewards = deque(maxlen=200)
        self.success_rates = deque(maxlen=200)
        self.epsilon_values = deque(maxlen=200)

        # Figure matplotlib pour generation d'images
        self.fig, self.ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        self.fig.patch.set_facecolor('#1a1a2e')

    def update_data(self, reward: float, success_rate: float, epsilon: float):
        """Met a jour les donnees"""
        self.rewards.append(reward)
        self.success_rates.append(success_rate)
        self.epsilon_values.append(epsilon)

    def render_rewards_plot(self) -> 'pygame.Surface':
        """Genere un plot des recompenses comme surface pygame"""
        import pygame

        self.ax.clear()
        self.ax.set_facecolor('#1a1a2e')

        if len(self.rewards) > 1:
            x = list(range(len(self.rewards)))
            self.ax.plot(x, list(self.rewards), color='#00ff88', alpha=0.5, linewidth=1)

            # Moyenne mobile
            if len(self.rewards) >= 20:
                avg = np.convolve(list(self.rewards), np.ones(20)/20, mode='valid')
                self.ax.plot(range(19, len(self.rewards)), avg, color='#ff6b6b', linewidth=2)

        self.ax.set_title('Recompenses', color='white', fontsize=10)
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.grid(True, alpha=0.2)

        # Convertir en surface pygame
        self.fig.canvas.draw()
        data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        return pygame.surfarray.make_surface(data.swapaxes(0, 1))

    def render_success_plot(self) -> 'pygame.Surface':
        """Genere un plot du taux de succes"""
        import pygame

        self.ax.clear()
        self.ax.set_facecolor('#1a1a2e')

        if len(self.success_rates) > 1:
            x = list(range(len(self.success_rates)))
            self.ax.fill_between(x, 0, list(self.success_rates), color='#4ecdc4', alpha=0.3)
            self.ax.plot(x, list(self.success_rates), color='#4ecdc4', linewidth=2)
            self.ax.axhline(y=80, color='#ff6b6b', linestyle='--', alpha=0.7)

        self.ax.set_title('Taux de Succes (%)', color='white', fontsize=10)
        self.ax.set_ylim(0, 100)
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.grid(True, alpha=0.2)

        self.fig.canvas.draw()
        data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        return pygame.surfarray.make_surface(data.swapaxes(0, 1))

    def render_epsilon_plot(self) -> 'pygame.Surface':
        """Genere un plot d'epsilon"""
        import pygame

        self.ax.clear()
        self.ax.set_facecolor('#1a1a2e')

        if len(self.epsilon_values) > 1:
            x = list(range(len(self.epsilon_values)))
            self.ax.plot(x, list(self.epsilon_values), color='#a855f7', linewidth=2)

        self.ax.set_title('Epsilon (Exploration)', color='white', fontsize=10)
        self.ax.set_ylim(0, 1)
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.grid(True, alpha=0.2)

        self.fig.canvas.draw()
        data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        return pygame.surfarray.make_surface(data.swapaxes(0, 1))

    def close(self):
        """Ferme la figure matplotlib"""
        plt.close(self.fig)


class LearningProgressBar:
    """
    Barre de progression de l'apprentissage pour pygame
    """

    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.epsilon = 1.0
        self.success_rate = 0.0
        self.episode = 0

    def update(self, epsilon: float, success_rate: float, episode: int):
        """Met a jour les valeurs"""
        self.epsilon = epsilon
        self.success_rate = success_rate
        self.episode = episode

    def draw(self, surface: 'pygame.Surface'):
        """Dessine la barre de progression"""
        import pygame

        # Fond
        pygame.draw.rect(surface, (30, 30, 50), (self.x, self.y, self.width, self.height))
        pygame.draw.rect(surface, (100, 100, 120), (self.x, self.y, self.width, self.height), 2)

        # Barre epsilon (exploration -> exploitation)
        exploration_progress = 1 - self.epsilon
        bar_width = int((self.width - 20) * exploration_progress)
        bar_color = self._lerp_color((255, 100, 100), (100, 255, 100), exploration_progress)
        pygame.draw.rect(surface, bar_color, (self.x + 10, self.y + 10, bar_width, 20))

        # Texte
        font = pygame.font.Font(None, 24)
        text = font.render(f"Apprentissage: {exploration_progress*100:.1f}%", True, (255, 255, 255))
        surface.blit(text, (self.x + 10, self.y + 35))

        # Taux de succes
        success_text = font.render(f"Succes: {self.success_rate:.1f}%", True, (100, 255, 150))
        surface.blit(success_text, (self.x + 10, self.y + 55))

        # Episode
        episode_text = font.render(f"Episode: {self.episode}", True, (200, 200, 200))
        surface.blit(episode_text, (self.x + 10, self.y + 75))

    def _lerp_color(self, c1, c2, t):
        """Interpolation lineaire entre deux couleurs"""
        return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


# Test du module
if __name__ == "__main__":
    print("Test des graphiques temps reel...")

    plots = RealtimeLearningPlots()
    plots.start()

    # Simulation de donnees
    import random
    epsilon = 1.0

    for episode in range(500):
        # Simulation d'un episode
        reward = random.gauss(-50 + episode * 0.3, 30)
        success = random.random() < (0.1 + episode * 0.0015)

        plots.log_episode(reward, success, epsilon)
        plots.log_loss(random.random() * 10 / (1 + episode * 0.01))
        plots.log_qvalue(random.gauss(episode * 0.1, 5))

        # Log de quelques etats et actions
        for _ in range(10):
            plots.log_state(random.uniform(0, 7000), random.uniform(0, 3000))
            plots.log_action(random.randint(-90, 90), random.randint(0, 4))

        epsilon *= 0.995
        time.sleep(0.05)

    print("Test termine. Fermez la fenetre pour quitter.")
    input("Appuyez sur Entree pour fermer...")
    plots.stop()
