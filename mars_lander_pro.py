# -*- coding: utf-8 -*-
"""
MARS LANDER - ULTIMATE PRO EDITION
===================================
Version ultime avec:
- PyTorch DQN (Deep Q-Network)
- Graphiques temps reel matplotlib
- Graphismes avances (particules, nebuleuses, terrain procedural)
- Effets visuels spectaculaires
- Visualisation complete de l'apprentissage IA

Auteur: Pierre Touzet
Idee originale: Florent Lannois
Assistance IA: Claude (Anthropic)
"""

import pygame
import sys
import math
import time
import random
import numpy as np
from typing import Optional, Tuple

# Imports des modules du projet
import data as data_module
from data import scenario0, scenario1, scenario2, scenario3, scenario4, scenario5
from data import fenX, fenY, echelle, alpha, gamma, epsilon, epsilon_decay
from vaisseau import Vaisseau
from surface import Surface
from jeu import Jeu
from affichage import Affichage
from ia_learning import IALearning

# Imports des modules PRO
from pytorch_ai import PyTorchDQNAgent
from pro_graphics import (
    SpaceBackground, AdvancedParticleSystem, AdvancedVesselRenderer,
    ProceduralTerrain, PostProcessor, SmoothCamera
)
from realtime_plots import RealtimeLearningPlots, LearningProgressBar

# Configuration
pygame.init()
pygame.mixer.init()

# Constantes
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
FPS = 60

# Couleurs
COLORS = {
    'background': (5, 5, 15),
    'text': (255, 255, 255),
    'success': (100, 255, 150),
    'danger': (255, 100, 100),
    'info': (100, 200, 255),
    'warning': (255, 200, 100),
    'accent': (168, 85, 247),
}


class ProGameStats:
    """Statistiques avancees du jeu"""

    def __init__(self):
        self.total_episodes = 0
        self.total_landings = 0
        self.total_crashes = 0
        self.current_streak = 0
        self.best_streak = 0
        self.total_reward = 0
        self.best_reward = float('-inf')
        self.start_time = time.time()

        # Historiques
        self.reward_history = []
        self.landing_positions = []
        self.crash_positions = []

    @property
    def success_rate(self) -> float:
        if self.total_episodes == 0:
            return 0
        return (self.total_landings / self.total_episodes) * 100

    @property
    def play_time(self) -> str:
        elapsed = int(time.time() - self.start_time)
        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class ProSoundManager:
    """Gestionnaire de sons avance"""

    def __init__(self):
        self.enabled = True
        self.sounds = {}
        self._load_sounds()

    def _load_sounds(self):
        """Charge ou genere les sons"""
        try:
            # Generation de sons synthetiques
            sample_rate = 44100

            # Son de propulsion (bruit continu)
            duration = 0.5
            t = np.linspace(0, duration, int(sample_rate * duration))
            thrust_wave = np.sin(2 * np.pi * 80 * t) * 0.3
            thrust_wave += np.random.randn(len(t)) * 0.1
            thrust_wave = (thrust_wave * 32767).astype(np.int16)
            thrust_stereo = np.column_stack((thrust_wave, thrust_wave))
            self.sounds['thrust'] = pygame.sndarray.make_sound(thrust_stereo)

            # Son d'explosion
            duration = 0.8
            t = np.linspace(0, duration, int(sample_rate * duration))
            explosion_wave = np.random.randn(len(t))
            envelope = np.exp(-t * 5)
            explosion_wave = explosion_wave * envelope * 0.5
            explosion_wave = (explosion_wave * 32767).astype(np.int16)
            explosion_stereo = np.column_stack((explosion_wave, explosion_wave))
            self.sounds['explosion'] = pygame.sndarray.make_sound(explosion_stereo)

            # Son de succes
            duration = 1.0
            t = np.linspace(0, duration, int(sample_rate * duration))
            freqs = [440, 554, 659, 880]
            success_wave = np.zeros_like(t)
            for i, freq in enumerate(freqs):
                start = int(len(t) * i / len(freqs))
                end = int(len(t) * (i + 1) / len(freqs))
                success_wave[start:end] = np.sin(2 * np.pi * freq * t[start:end]) * 0.3
            envelope = np.exp(-t * 2)
            success_wave = success_wave * envelope
            success_wave = (success_wave * 32767).astype(np.int16)
            success_stereo = np.column_stack((success_wave, success_wave))
            self.sounds['success'] = pygame.sndarray.make_sound(success_stereo)

            print("[Audio] Sons generes avec succes")
        except Exception as e:
            print(f"[Audio] Erreur generation sons: {e}")
            self.enabled = False

    def play(self, sound_name: str, volume: float = 0.5):
        """Joue un son"""
        if not self.enabled or sound_name not in self.sounds:
            return
        try:
            sound = self.sounds[sound_name]
            sound.set_volume(volume)
            sound.play()
        except:
            pass

    def stop(self, sound_name: str):
        """Arrete un son"""
        if sound_name in self.sounds:
            try:
                self.sounds[sound_name].stop()
            except:
                pass


class MarsLanderPro:
    """
    Jeu Mars Lander ULTIMATE PRO
    """

    def __init__(self):
        print("=" * 60)
        print("  MARS LANDER - ULTIMATE PRO EDITION")
        print("  PyTorch DQN + Graphiques Temps Reel")
        print("=" * 60)

        # Fenetre principale
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Mars Lander - ULTIMATE PRO Edition")
        self.clock = pygame.time.Clock()

        # Etat du jeu
        self.running = True
        self.paused = False
        self.show_plots = False
        self.show_debug = True
        self.use_pytorch = True
        self.simulation_speed = 1

        # Statistiques
        self.stats = ProGameStats()

        # Sons
        self.sound = ProSoundManager()

        # Graphismes PRO
        self._init_graphics()

        # IA
        self._init_ai()

        # Jeu
        self._init_game()

        # Plots temps reel (optionnel)
        self.realtime_plots = None
        self.progress_bar = LearningProgressBar(WINDOW_WIDTH - 220, 10, 210, 100)

        print("[Game] Initialisation complete!")

    def _init_graphics(self):
        """Initialise les graphismes avances"""
        print("[Graphics] Initialisation...")

        self.background = SpaceBackground(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.particles = AdvancedParticleSystem(3000)
        self.vessel_renderer = AdvancedVesselRenderer()
        self.post_processor = PostProcessor(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.camera = SmoothCamera(WINDOW_WIDTH, WINDOW_HEIGHT, 7000, 3000)

        # Terrain procedural
        self.procedural_terrain = ProceduralTerrain(7000)

        # Polices
        self.fonts = {
            'title': pygame.font.Font(None, 48),
            'large': pygame.font.Font(None, 36),
            'medium': pygame.font.Font(None, 28),
            'small': pygame.font.Font(None, 22),
            'tiny': pygame.font.Font(None, 18),
        }

        print("[Graphics] OK")

    def _init_ai(self):
        """Initialise les systemes d'IA"""
        print("[AI] Initialisation...")

        # PyTorch DQN Agent
        self.pytorch_agent = PyTorchDQNAgent(
            state_size=8,
            n_angles=37,
            n_powers=5,
            hidden_size=256,
            learning_rate=0.0003,
            gamma=0.99,
            epsilon_start=0.9,
            epsilon_end=0.01,
            epsilon_decay=0.9997,
            batch_size=64
        )

        # Tentative de chargement d'un modele existant
        self.pytorch_agent.load("historique/pytorch_model.pth")

        print(f"[AI] PyTorch device: {self.pytorch_agent.get_stats()['device']}")
        print("[AI] OK")

    def _init_game(self):
        """Initialise le jeu"""
        print("[Game] Initialisation du scenario...")

        # Scenario actuel
        self.scenar = scenario0
        self.scenarios = [scenario0, scenario1, scenario2, scenario3, scenario4, scenario5]
        self.current_scenario = 0

        # Parametres d'affichage
        self.echelle = echelle
        self.fenX = fenX
        self.fenY = fenY

        # Initialisation des composants
        self.vessel = Vaisseau()
        self.vessel.init_vaisseau(self.scenar['vaisseau'])

        self.surface = Surface(self.scenar['surface_mars'])
        self.affichage = Affichage()
        self.jeu = Jeu(self.scenar)

        # Zone d'atterrissage
        self.zone = self.surface.calcul_zone_atterissage(self.scenar)
        self.affichage.init_terrain(self.scenar['surface_mars'], self.zone)

        # IA Q-Learning classique (backup)
        toutes_actions = self.jeu.toutes_actions_possibles(self.vessel)
        self.ia_classic = IALearning(self.scenar, toutes_actions, alpha, gamma, epsilon, epsilon_decay, True)

        # Initialisation des variables d'episode
        self.episode_reward = 0
        self.episode_steps = 0
        self.terminal_time = None

        print("[Game] OK")

    def _reset_vessel(self):
        """Reinitialise le vaisseau"""
        self.vessel = Vaisseau()
        self.vessel.init_vaisseau(self.scenar['vaisseau'])

        self.episode_reward = 0
        self.episode_steps = 0
        self.terminal_time = None

    def _get_state(self) -> Tuple:
        """Recupere l'etat actuel pour l'IA"""
        # Position
        x = self.vessel.x
        y = self.vessel.y

        # Vitesses
        h_speed = self.vessel.h_speed
        v_speed = self.vessel.v_speed

        # Fuel et angle
        fuel = self.vessel.fuel
        angle = self.vessel.angle

        # Distance a la zone d'atterrissage
        zone = self.surface.atterissage
        zone_x1 = zone[0][0]
        zone_x2 = zone[1][0]
        zone_y = zone[0][1]
        zone_center = (zone_x1 + zone_x2) / 2

        dist_zone = abs(x - zone_center)
        alt_zone = y - zone_y

        return (x, y, h_speed, v_speed, fuel, angle, dist_zone, alt_zone)

    def _calculate_reward(self) -> float:
        """Calcule la recompense"""
        reward = 0

        # Position par rapport a la zone
        x = self.vessel.x
        y = self.vessel.y
        zone = self.surface.atterissage
        zone_x1 = zone[0][0]
        zone_x2 = zone[1][0]
        zone_center = (zone_x1 + zone_x2) / 2

        # Reward shaping
        dist_to_zone = abs(x - zone_center)
        reward -= dist_to_zone * 0.001  # Penalite pour distance

        # Bonus pour etre au-dessus de la zone
        if zone_x1 <= x <= zone_x2:
            reward += 1

        # Penalite pour vitesse excessive
        speed = math.sqrt(self.vessel.h_speed**2 + self.vessel.v_speed**2)
        if speed > 50:
            reward -= speed * 0.01

        # Bonus/Malus terminal
        if self.vessel.est_pose:
            reward += 1000
            # Bonus pour atterrissage en douceur
            if abs(self.vessel.v_speed) < 20:
                reward += 500
            if abs(self.vessel.h_speed) < 10:
                reward += 300
        elif self.vessel.detruit:
            reward -= 500

        return reward

    def _handle_events(self):
        """Gere les evenements"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

                elif event.key == pygame.K_SPACE:
                    self._restart_episode()

                elif event.key == pygame.K_p:
                    self.paused = not self.paused

                elif event.key == pygame.K_g:
                    # Toggle graphiques temps reel
                    if self.realtime_plots is None:
                        self.realtime_plots = RealtimeLearningPlots()
                        self.realtime_plots.start()
                        print("[Plots] Fenetre graphiques ouverte")
                    else:
                        self.realtime_plots.stop()
                        self.realtime_plots = None
                        print("[Plots] Fenetre graphiques fermee")

                elif event.key == pygame.K_d:
                    self.show_debug = not self.show_debug

                elif event.key == pygame.K_t:
                    self.use_pytorch = not self.use_pytorch
                    print(f"[AI] Mode: {'PyTorch DQN' if self.use_pytorch else 'Q-Learning classique'}")

                elif event.key == pygame.K_s:
                    self._save_model()

                elif event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS:
                    self.simulation_speed = min(10, self.simulation_speed + 1)

                elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                    self.simulation_speed = max(1, self.simulation_speed - 1)

                # Scenarios F1-F6
                elif event.key in [pygame.K_F1, pygame.K_F2, pygame.K_F3, pygame.K_F4, pygame.K_F5, pygame.K_F6]:
                    scenario_num = event.key - pygame.K_F1
                    self._change_scenario(scenario_num)

    def _change_scenario(self, num: int):
        """Change le scenario"""
        if num < len(self.scenarios):
            self.current_scenario = num
            self.scenar = self.scenarios[num]
            self.surface = Surface(self.scenar['surface_mars'])
            self.zone = self.surface.calcul_zone_atterissage(self.scenar)
            self.affichage.init_terrain(self.scenar['surface_mars'], self.zone)
            self.jeu = Jeu(self.scenar)
            self._reset_vessel()
            print(f"[Game] Scenario {num + 1} charge")

    def _restart_episode(self):
        """Redemarre un episode"""
        self._reset_vessel()
        self.stats.total_episodes += 1

    def _save_model(self):
        """Sauvegarde le modele"""
        self.pytorch_agent.save("historique/pytorch_model.pth")
        self.ia_classic.save_q_table()
        print("[Save] Modeles sauvegardes")

    def _update(self):
        """Mise a jour du jeu"""
        if self.paused:
            return

        for _ in range(self.simulation_speed):
            self._update_step()

    def _update_step(self):
        """Une etape de mise a jour"""
        if self.vessel.detruit or self.vessel.est_pose:
            # Gestion fin d'episode
            if self.terminal_time is None:
                self.terminal_time = time.time()

                # Calcul recompense finale
                final_reward = self._calculate_reward()
                self.episode_reward += final_reward

                # Mise a jour stats
                success = self.vessel.est_pose

                # Coordonnees ecran pour particules
                px = self.vessel.x * WINDOW_WIDTH / self.fenX
                py = WINDOW_HEIGHT - (self.vessel.y * WINDOW_HEIGHT / self.fenY)

                if success:
                    self.stats.total_landings += 1
                    self.stats.current_streak += 1
                    self.stats.best_streak = max(self.stats.best_streak, self.stats.current_streak)
                    self.stats.landing_positions.append((self.vessel.x, self.vessel.y))
                    self.sound.play('success')
                    self.particles.emit(px, py, 'celebration', 100)
                else:
                    self.stats.total_crashes += 1
                    self.stats.current_streak = 0
                    self.stats.crash_positions.append((self.vessel.x, self.vessel.y))
                    self.sound.play('explosion')
                    self.particles.emit(px, py, 'explosion', 80)

                self.stats.reward_history.append(self.episode_reward)
                self.stats.best_reward = max(self.stats.best_reward, self.episode_reward)

                # Log aux plots
                if self.realtime_plots:
                    epsilon = self.pytorch_agent.epsilon if self.use_pytorch else self.ia_classic.epsilon
                    self.realtime_plots.log_episode(self.episode_reward, success, epsilon)

            # Auto-restart apres delai
            if time.time() - self.terminal_time > 1.5:
                self._restart_episode()

            return

        # Recuperer l'etat
        state = self._get_state()

        # Choisir l'action
        if self.use_pytorch:
            action = self.pytorch_agent.choose_action(state)
        else:
            q_state = self.ia_classic.recupere_etat(self.vessel, self.surface)
            action = self.ia_classic.choisir_action(q_state)

        # Appliquer l'action
        if action:
            self.vessel.angle, self.vessel.puissance = action

            # Log action
            if self.realtime_plots:
                self.realtime_plots.log_action(action[0], action[1])

        # Physique
        self.jeu.actualisation(self.vessel, self.affichage, self.surface, self.ia_classic)

        # Detection collision
        self.jeu.touche_mars(self.affichage, self.vessel, self.surface)

        # Nouvel etat et recompense
        next_state = self._get_state()
        reward = self._calculate_reward()
        self.episode_reward += reward
        done = self.vessel.detruit or self.vessel.est_pose

        # Apprentissage
        if self.use_pytorch:
            self.pytorch_agent.store_experience(state, action, reward, next_state, done)
            self.pytorch_agent.train_step()
            self.pytorch_agent.decay_epsilon()

            # Log loss et Q-value
            if self.realtime_plots:
                stats = self.pytorch_agent.get_stats()
                if stats['avg_loss'] > 0:
                    self.realtime_plots.log_loss(stats['avg_loss'])
                if stats['avg_q'] != 0:
                    self.realtime_plots.log_qvalue(stats['avg_q'])
        else:
            q_state = self.ia_classic.recupere_etat(self.vessel, self.surface)
            q_next_state = self.ia_classic.recupere_etat(self.vessel, self.surface)
            self.ia_classic.update_q_table(q_state, action, reward, q_next_state, done)
            self.ia_classic.decay_epsilon()

        # Log etat
        if self.realtime_plots:
            self.realtime_plots.log_state(state[0], state[1])

        # Particules de propulsion
        if self.vessel.puissance > 0 and not self.vessel.detruit:
            px = self.vessel.x * WINDOW_WIDTH / self.fenX
            py = WINDOW_HEIGHT - (self.vessel.y * WINDOW_HEIGHT / self.fenY)
            self.particles.emit(px, py + 15, 'flame', self.vessel.puissance * 2,
                              direction=math.radians(self.vessel.angle + 90),
                              spread=math.pi/6)

        self.episode_steps += 1

    def _render(self):
        """Rendu graphique"""
        # Fond spatial
        self.background.update()
        self.background.draw(self.screen)

        # Terrain
        self._draw_terrain()

        # Zone d'atterrissage
        self._draw_landing_zone()

        # Particules
        self.particles.update()
        self.particles.draw(self.screen)

        # Vaisseau
        if not self.vessel.detruit:
            # Convertir en coordonnees ecran
            vx = self.vessel.x * WINDOW_WIDTH / self.fenX
            vy = WINDOW_HEIGHT - (self.vessel.y * WINDOW_HEIGHT / self.fenY)
            self.vessel_renderer.draw(
                self.screen, vx, vy, self.vessel.angle,
                self.vessel.puissance, damage=0, scale=1.2
            )

        # Interface
        self._draw_ui()

        # Debug info
        if self.show_debug:
            self._draw_debug()

        # Progress bar
        epsilon = self.pytorch_agent.epsilon if self.use_pytorch else self.ia_classic.epsilon
        self.progress_bar.update(epsilon, self.stats.success_rate, self.stats.total_episodes)
        self.progress_bar.draw(self.screen)

        pygame.display.flip()

    def _draw_terrain(self):
        """Dessine le terrain"""
        # Utiliser les points de la surface mars
        terrain_points = []
        for x, y in self.surface.mars_surface:
            # Convertir en coordonnees ecran
            screen_x = x * WINDOW_WIDTH / self.fenX
            screen_y = WINDOW_HEIGHT - (y * WINDOW_HEIGHT / self.fenY)
            terrain_points.append((screen_x, screen_y))

        if len(terrain_points) > 1:
            # Ajouter les coins pour le remplissage
            fill_points = terrain_points + [(WINDOW_WIDTH, WINDOW_HEIGHT), (0, WINDOW_HEIGHT)]
            pygame.draw.polygon(self.screen, (60, 40, 30), fill_points)

            # Contour
            pygame.draw.lines(self.screen, (150, 100, 70), False, terrain_points, 3)

    def _draw_landing_zone(self):
        """Dessine la zone d'atterrissage"""
        zone = self.surface.atterissage
        if not zone:
            return

        # Coordonnees de la zone
        zone_x1, zone_y = zone[0]
        zone_x2, _ = zone[1]

        # Convertir en coordonnees ecran
        screen_x1 = zone_x1 * WINDOW_WIDTH / self.fenX
        screen_x2 = zone_x2 * WINDOW_WIDTH / self.fenX
        screen_y = WINDOW_HEIGHT - (zone_y * WINDOW_HEIGHT / self.fenY)

        # Zone verte pulsante
        pulse = (math.sin(time.time() * 3) + 1) / 2
        color = (int(50 + pulse * 50), int(200 + pulse * 55), int(100 + pulse * 50))

        pygame.draw.line(self.screen, color, (screen_x1, screen_y), (screen_x2, screen_y), 5)

        # Marqueurs
        for x in [screen_x1, screen_x2]:
            pygame.draw.circle(self.screen, color, (int(x), int(screen_y)), 8)

    def _draw_ui(self):
        """Dessine l'interface"""
        # Panel gauche - Info vaisseau
        panel_x = 10
        panel_y = 10

        # Fond semi-transparent
        panel = pygame.Surface((200, 180), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 150))
        self.screen.blit(panel, (panel_x, panel_y))

        # Titre
        title = self.fonts['medium'].render("VAISSEAU", True, COLORS['accent'])
        self.screen.blit(title, (panel_x + 10, panel_y + 10))

        # Infos
        y_offset = 40
        infos = [
            (f"Altitude: {self.vessel.y / self.echelle:.0f} m", COLORS['text']),
            (f"H-Speed: {self.vessel.h_speed:.1f} m/s", COLORS['warning'] if abs(self.vessel.h_speed) > 20 else COLORS['success']),
            (f"V-Speed: {self.vessel.v_speed:.1f} m/s", COLORS['danger'] if self.vessel.v_speed < -40 else COLORS['success']),
            (f"Fuel: {self.vessel.fuel:.0f} L", COLORS['danger'] if self.vessel.fuel < 200 else COLORS['text']),
            (f"Angle: {self.vessel.angle}*", COLORS['warning'] if self.vessel.angle != 0 else COLORS['success']),
            (f"Power: {self.vessel.puissance}", COLORS['info']),
        ]

        for text, color in infos:
            surf = self.fonts['small'].render(text, True, color)
            self.screen.blit(surf, (panel_x + 10, panel_y + y_offset))
            y_offset += 22

        # Panel statistiques
        stats_x = 10
        stats_y = 200

        stats_panel = pygame.Surface((200, 150), pygame.SRCALPHA)
        stats_panel.fill((0, 0, 0, 150))
        self.screen.blit(stats_panel, (stats_x, stats_y))

        title = self.fonts['medium'].render("STATS", True, COLORS['accent'])
        self.screen.blit(title, (stats_x + 10, stats_y + 10))

        y_offset = 40
        stats_info = [
            (f"Episodes: {self.stats.total_episodes}", COLORS['text']),
            (f"Succes: {self.stats.success_rate:.1f}%", COLORS['success'] if self.stats.success_rate > 50 else COLORS['danger']),
            (f"Serie: {self.stats.current_streak} (max: {self.stats.best_streak})", COLORS['info']),
            (f"Temps: {self.stats.play_time}", COLORS['text']),
        ]

        for text, color in stats_info:
            surf = self.fonts['small'].render(text, True, color)
            self.screen.blit(surf, (stats_x + 10, stats_y + y_offset))
            y_offset += 22

        # Controles en bas
        controls = [
            "ESPACE: Restart | P: Pause | G: Graphiques | D: Debug",
            "T: Toggle IA | S: Save | +/-: Vitesse | F1-F6: Scenarios"
        ]

        for i, text in enumerate(controls):
            surf = self.fonts['tiny'].render(text, True, (150, 150, 150))
            self.screen.blit(surf, (10, WINDOW_HEIGHT - 40 + i * 18))

    def _draw_debug(self):
        """Dessine les informations de debug"""
        debug_x = WINDOW_WIDTH - 220
        debug_y = 120

        debug_panel = pygame.Surface((210, 200), pygame.SRCALPHA)
        debug_panel.fill((0, 0, 0, 150))
        self.screen.blit(debug_panel, (debug_x, debug_y))

        title = self.fonts['medium'].render("DEBUG IA", True, COLORS['accent'])
        self.screen.blit(title, (debug_x + 10, debug_y + 10))

        if self.use_pytorch:
            stats = self.pytorch_agent.get_stats()
            info = [
                f"Mode: PyTorch DQN",
                f"Device: {stats['device']}",
                f"Epsilon: {stats['epsilon']:.4f}",
                f"Buffer: {stats['buffer_size']}",
                f"Avg Loss: {stats['avg_loss']:.4f}",
                f"Avg Q: {stats['avg_q']:.2f}",
            ]
        else:
            info = [
                f"Mode: Q-Learning",
                f"Epsilon: {self.ia_classic.epsilon:.4f}",
                f"Q-Table: {len(self.ia_classic.q_table)} etats",
            ]

        info.append(f"Sim Speed: x{self.simulation_speed}")
        info.append(f"FPS: {self.clock.get_fps():.0f}")

        y_offset = 40
        for text in info:
            surf = self.fonts['tiny'].render(text, True, COLORS['text'])
            self.screen.blit(surf, (debug_x + 10, debug_y + y_offset))
            y_offset += 18

    def run(self):
        """Boucle principale"""
        print("\n[Game] Demarrage...")
        print("Controles:")
        print("  G - Ouvrir graphiques temps reel")
        print("  T - Toggle PyTorch/Q-Learning")
        print("  S - Sauvegarder modele")
        print("  +/- - Vitesse simulation")
        print()

        while self.running:
            self._handle_events()
            self._update()
            self._render()
            self.clock.tick(FPS)

        # Cleanup
        print("\n[Game] Fermeture...")
        self._save_model()

        if self.realtime_plots:
            self.realtime_plots.stop()

        pygame.quit()
        print("[Game] Au revoir!")


def main():
    """Point d'entree"""
    game = MarsLanderPro()
    game.run()


if __name__ == "__main__":
    main()
