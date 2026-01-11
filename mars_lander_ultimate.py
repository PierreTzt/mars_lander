"""
MARS LANDER ULTIMATE - Experience IA Memorable

Une experience visuelle immersive montrant l'apprentissage de l'IA en temps reel.
Effets visuels avances, feedback immediat, et progression visible.

Par Florent Lannois (idee originale) & Claude AI
"""

import os
import sys
import time
import random
import math
import pygame
from collections import deque

# Imports locaux
from data import *
import data as data_module
from vaisseau import Vaisseau
from jeu import Jeu
from affichage import Affichage
from surface import Surface
from ia_learning import IALearning
from mars_lander import WindSystem, SoundManager


# =============================================================================
# EFFETS VISUELS AVANCES
# =============================================================================

class ScreenShake:
    """Effet de tremblement d'ecran."""
    def __init__(self):
        self.intensity = 0
        self.duration = 0
        self.offset_x = 0
        self.offset_y = 0

    def shake(self, intensity: float = 10, duration: float = 0.5):
        self.intensity = intensity
        self.duration = duration

    def update(self, dt: float):
        if self.duration > 0:
            self.duration -= dt
            self.offset_x = random.uniform(-self.intensity, self.intensity)
            self.offset_y = random.uniform(-self.intensity, self.intensity)
            self.intensity *= 0.9
        else:
            self.offset_x = 0
            self.offset_y = 0

    def get_offset(self):
        return (int(self.offset_x), int(self.offset_y))


class CelebrationEffect:
    """Effet de celebration lors d'un atterrissage reussi."""
    def __init__(self, screen_width, screen_height):
        self.width = screen_width
        self.height = screen_height
        self.particles = []
        self.active = False
        self.duration = 0

    def trigger(self, x: float, y: float):
        self.active = True
        self.duration = 3.0
        # Creer des particules de feu d'artifice
        colors = [(255, 215, 0), (0, 255, 100), (100, 200, 255), (255, 100, 100), (255, 255, 255)]
        for _ in range(100):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 8)
            self.particles.append({
                'x': x, 'y': y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed - 3,
                'color': random.choice(colors),
                'size': random.uniform(2, 5),
                'life': random.uniform(1, 2.5),
                'gravity': 0.15
            })

    def update(self, dt: float):
        if not self.active:
            return

        self.duration -= dt
        if self.duration <= 0:
            self.active = False
            self.particles = []
            return

        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += p['gravity']
            p['life'] -= dt
            p['size'] *= 0.98

        self.particles = [p for p in self.particles if p['life'] > 0 and p['size'] > 0.5]

    def draw(self, screen):
        for p in self.particles:
            alpha = min(255, int(p['life'] * 200))
            color = (*p['color'][:3], alpha)
            surf = pygame.Surface((int(p['size']*2), int(p['size']*2)), pygame.SRCALPHA)
            pygame.draw.circle(surf, color, (int(p['size']), int(p['size'])), int(p['size']))
            screen.blit(surf, (int(p['x'] - p['size']), int(p['y'] - p['size'])))


class AIProgressDisplay:
    """Affichage avance de la progression de l'IA."""
    def __init__(self, screen_width, screen_height):
        self.width = screen_width
        self.height = screen_height

        # Historique
        self.rewards_history = deque(maxlen=500)
        self.success_history = deque(maxlen=100)  # 1 pour succes, 0 pour echec
        self.episode_times = deque(maxlen=100)

        # Stats
        self.total_episodes = 0
        self.total_successes = 0
        self.best_reward = float('-inf')
        self.current_streak = 0
        self.best_streak = 0
        self.start_time = time.time()

        # Animation
        self.pulse = 0
        self.last_success = False
        self.success_flash = 0

    def record_episode(self, reward: float, success: bool, duration: float):
        self.rewards_history.append(reward)
        self.success_history.append(1 if success else 0)
        self.episode_times.append(duration)
        self.total_episodes += 1

        if success:
            self.total_successes += 1
            self.current_streak += 1
            self.best_streak = max(self.best_streak, self.current_streak)
            self.success_flash = 1.0
            self.last_success = True
        else:
            self.current_streak = 0
            self.last_success = False

        if reward > self.best_reward:
            self.best_reward = reward

    def get_success_rate(self) -> float:
        if not self.success_history:
            return 0
        return sum(self.success_history) / len(self.success_history)

    def get_avg_reward(self, n: int = 50) -> float:
        if not self.rewards_history:
            return 0
        recent = list(self.rewards_history)[-n:]
        return sum(recent) / len(recent)

    def update(self, dt: float):
        self.pulse = (self.pulse + dt * 3) % (2 * math.pi)
        self.success_flash = max(0, self.success_flash - dt * 2)

    def draw(self, screen, font, ia):
        # Couleurs
        bg_color = (10, 15, 30, 220)
        accent = (0, 200, 255)
        success_color = (0, 255, 100)
        fail_color = (255, 80, 80)

        # Panel principal en haut a droite
        panel_width = 280
        panel_height = 200
        panel_x = self.width - panel_width - 10
        panel_y = 10

        # Fond du panel avec bordure animee
        panel_surf = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surf.fill(bg_color)

        # Bordure avec effet pulse
        border_intensity = int(100 + 50 * math.sin(self.pulse))
        border_color = (0, border_intensity, 200)
        pygame.draw.rect(panel_surf, border_color, (0, 0, panel_width, panel_height), 2)

        # Flash de succes
        if self.success_flash > 0:
            flash_surf = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
            flash_alpha = int(self.success_flash * 100)
            flash_surf.fill((*success_color, flash_alpha))
            panel_surf.blit(flash_surf, (0, 0))

        screen.blit(panel_surf, (panel_x, panel_y))

        y = panel_y + 10

        # Titre
        title = font.render("PROGRESSION IA", True, accent)
        screen.blit(title, (panel_x + panel_width//2 - title.get_width()//2, y))
        y += 25

        # Episode
        ep_text = font.render(f"Episode: {self.total_episodes}", True, (255, 255, 255))
        screen.blit(ep_text, (panel_x + 10, y))
        y += 20

        # Taux de reussite avec barre de progression
        success_rate = self.get_success_rate()
        rate_text = font.render(f"Taux: {success_rate:.1%}", True,
                               success_color if success_rate > 0.5 else fail_color)
        screen.blit(rate_text, (panel_x + 10, y))

        # Barre de progression
        bar_x = panel_x + 100
        bar_width = 160
        bar_height = 12
        pygame.draw.rect(screen, (50, 50, 50), (bar_x, y + 2, bar_width, bar_height))
        fill_width = int(bar_width * success_rate)
        bar_color = success_color if success_rate > 0.5 else (255, 200, 0) if success_rate > 0.2 else fail_color
        pygame.draw.rect(screen, bar_color, (bar_x, y + 2, fill_width, bar_height))
        pygame.draw.rect(screen, (100, 100, 100), (bar_x, y + 2, bar_width, bar_height), 1)
        y += 22

        # Serie actuelle
        streak_color = success_color if self.current_streak >= 3 else (255, 255, 255)
        streak_text = font.render(f"Serie: {self.current_streak} (Best: {self.best_streak})", True, streak_color)
        screen.blit(streak_text, (panel_x + 10, y))
        y += 20

        # Recompense moyenne
        avg_reward = self.get_avg_reward()
        reward_color = success_color if avg_reward > 0 else fail_color
        reward_text = font.render(f"Reward Moy: {avg_reward:.1f}", True, reward_color)
        screen.blit(reward_text, (panel_x + 10, y))
        y += 20

        # Epsilon (exploration)
        epsilon = ia.epsilon if hasattr(ia, 'epsilon') else 0
        eps_text = font.render(f"Exploration: {epsilon:.2%}", True, (200, 200, 200))
        screen.blit(eps_text, (panel_x + 10, y))

        # Barre epsilon
        eps_bar_x = panel_x + 120
        eps_bar_width = 140
        pygame.draw.rect(screen, (50, 50, 50), (eps_bar_x, y + 2, eps_bar_width, 10))
        pygame.draw.rect(screen, (255, 200, 0), (eps_bar_x, y + 2, int(eps_bar_width * epsilon), 10))
        y += 20

        # Q-Table size
        q_size = len(ia.q_table) if hasattr(ia, 'q_table') else 0
        q_text = font.render(f"Q-Table: {q_size:,} etats", True, (150, 150, 200))
        screen.blit(q_text, (panel_x + 10, y))
        y += 20

        # Temps ecoule
        elapsed = time.time() - self.start_time
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        time_text = font.render(f"Temps: {mins:02d}:{secs:02d}", True, (150, 150, 150))
        screen.blit(time_text, (panel_x + 10, y))

        # Mini graphique de progression
        self._draw_mini_graph(screen, panel_x + 10, panel_y + panel_height + 10, panel_width - 20, 60)

    def _draw_mini_graph(self, screen, x, y, width, height):
        """Dessine un mini graphique des recompenses."""
        if len(self.rewards_history) < 2:
            return

        # Fond
        pygame.draw.rect(screen, (20, 25, 40), (x, y, width, height))
        pygame.draw.rect(screen, (50, 60, 80), (x, y, width, height), 1)

        # Ligne zero
        zero_y = y + height // 2
        pygame.draw.line(screen, (60, 60, 60), (x, zero_y), (x + width, zero_y), 1)

        # Points
        data = list(self.rewards_history)
        if not data:
            return

        min_val = min(data)
        max_val = max(data)
        if max_val == min_val:
            max_val = min_val + 1

        points = []
        for i, val in enumerate(data):
            px = x + (i / max(1, len(data) - 1)) * width
            py = y + height - ((val - min_val) / (max_val - min_val)) * height
            points.append((px, py))

        # Dessiner la ligne avec gradient de couleur
        if len(points) >= 2:
            for i in range(len(points) - 1):
                # Couleur basee sur la valeur
                val = data[i]
                if val > 0:
                    color = (0, 200, 100)
                elif val > -20:
                    color = (200, 200, 0)
                else:
                    color = (200, 80, 80)
                pygame.draw.line(screen, color, points[i], points[i + 1], 2)


class ExplosionEffect:
    """Effet d'explosion ameliore."""
    def __init__(self):
        self.particles = []
        self.debris = []
        self.active = False

    def trigger(self, x: float, y: float):
        self.active = True

        # Particules de feu
        for _ in range(80):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 6)
            self.particles.append({
                'x': x, 'y': y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed - 2,
                'color': random.choice([(255, 200, 50), (255, 150, 30), (255, 80, 20), (255, 50, 10)]),
                'size': random.uniform(3, 8),
                'life': random.uniform(0.5, 1.5)
            })

        # Debris
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 5)
            self.debris.append({
                'x': x, 'y': y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed - 3,
                'angle': random.uniform(0, 360),
                'rot_speed': random.uniform(-10, 10),
                'size': random.uniform(3, 8),
                'life': random.uniform(1, 2)
            })

    def update(self, dt: float):
        if not self.active:
            return

        # Particules
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.2  # Gravite
            p['life'] -= dt
            p['size'] *= 0.97

        # Debris
        for d in self.debris:
            d['x'] += d['vx']
            d['y'] += d['vy']
            d['vy'] += 0.3
            d['angle'] += d['rot_speed']
            d['life'] -= dt

        self.particles = [p for p in self.particles if p['life'] > 0]
        self.debris = [d for d in self.debris if d['life'] > 0]

        if not self.particles and not self.debris:
            self.active = False

    def draw(self, screen):
        # Particules de feu
        for p in self.particles:
            alpha = min(255, int(p['life'] * 255))
            surf = pygame.Surface((int(p['size']*2), int(p['size']*2)), pygame.SRCALPHA)
            color = (*p['color'], alpha)
            pygame.draw.circle(surf, color, (int(p['size']), int(p['size'])), int(p['size']))
            screen.blit(surf, (int(p['x'] - p['size']), int(p['y'] - p['size'])))

        # Debris
        for d in self.debris:
            surf = pygame.Surface((int(d['size']*2), int(d['size']*2)), pygame.SRCALPHA)
            pygame.draw.polygon(surf, (100, 80, 60, 200), [
                (d['size'], 0),
                (d['size']*2, d['size']),
                (d['size'], d['size']*2),
                (0, d['size'])
            ])
            rotated = pygame.transform.rotate(surf, d['angle'])
            screen.blit(rotated, (int(d['x'] - rotated.get_width()//2),
                                 int(d['y'] - rotated.get_height()//2)))


class BackgroundStars:
    """Fond etoile anime."""
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.stars = []

        # Generer les etoiles
        for _ in range(200):
            self.stars.append({
                'x': random.randint(0, width),
                'y': random.randint(0, height),
                'size': random.uniform(0.5, 2),
                'brightness': random.uniform(0.3, 1),
                'twinkle_speed': random.uniform(1, 3),
                'twinkle_offset': random.uniform(0, 2 * math.pi)
            })

    def draw(self, screen, time_offset: float):
        for star in self.stars:
            # Scintillement
            twinkle = 0.5 + 0.5 * math.sin(time_offset * star['twinkle_speed'] + star['twinkle_offset'])
            brightness = int(255 * star['brightness'] * twinkle)
            color = (brightness, brightness, min(255, brightness + 20))

            if star['size'] > 1.5:
                pygame.draw.circle(screen, color, (int(star['x']), int(star['y'])), int(star['size']))
            else:
                screen.set_at((int(star['x']), int(star['y'])), color)


class LandingTrail:
    """Trace du chemin de descente."""
    def __init__(self, max_points=500):
        self.points = deque(maxlen=max_points)
        self.active = True

    def add_point(self, x: float, y: float, speed: float):
        self.points.append({'x': x, 'y': y, 'speed': speed})

    def clear(self):
        self.points.clear()

    def draw(self, screen, echelle):
        if len(self.points) < 2:
            return

        points_list = list(self.points)
        for i in range(1, len(points_list)):
            p1 = points_list[i-1]
            p2 = points_list[i]

            # Couleur basee sur la vitesse
            speed = p2['speed']
            if speed < 30:
                color = (0, 200, 100)  # Vert - bonne vitesse
            elif speed < 50:
                color = (200, 200, 0)  # Jaune - attention
            else:
                color = (200, 50, 50)  # Rouge - trop vite

            x1, y1 = int(p1['x'] / echelle), int(p1['y'] / echelle)
            x2, y2 = int(p2['x'] / echelle), int(p2['y'] / echelle)

            # Fade out progressif
            alpha = int(255 * (i / len(points_list)))

            pygame.draw.line(screen, color, (x1, y1), (x2, y2), 1)


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def main():
    """Point d'entree principal."""

    # Initialisation
    pygame.init()
    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

    # Ecran
    screen_width = fenX // echelle
    screen_height = fenY // echelle
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("MARS LANDER ULTIMATE - IA Learning Experience")

    # Polices
    font_small = pygame.font.Font(None, 18)
    font = pygame.font.Font(None, 24)
    font_large = pygame.font.Font(None, 36)
    font_title = pygame.font.Font(None, 48)

    # Composants de base
    scenar = scenario0
    v = Vaisseau()
    v.init_vaisseau(scenar['vaisseau'])
    s = Surface(scenar['surface_mars'])
    a = Affichage()
    j = Jeu(scenar)

    zone = s.calcul_zone_atterissage(scenar)
    a.init_terrain(scenar['surface_mars'], zone)
    a.set_landing_zone(zone)

    toutes_actions = j.toutes_actions_possibles(v)
    ia = IALearning(scenar, toutes_actions, alpha, gamma, epsilon, epsilon_decay, ia_active)

    # Effets visuels
    screen_shake = ScreenShake()
    celebration = CelebrationEffect(screen_width, screen_height)
    explosion = ExplosionEffect()
    stars = BackgroundStars(screen_width, screen_height)
    progress_display = AIProgressDisplay(screen_width, screen_height)
    trail = LandingTrail()

    # Systemes
    wind = WindSystem()
    sound = SoundManager()

    # Variables
    clock = pygame.time.Clock()
    running = True

    speed_levels = [30, 60, 120, 500, 1000, 5000]
    speed_index = 1
    current_speed = speed_levels[speed_index]

    last_destroyed = False
    last_landed = False
    terminal_time = None
    LANDING_DELAY = 2.0
    episode_start_time = time.time()
    global_time = 0

    paused = False
    show_trail = True

    # Message d'accueil
    print("\n" + "="*60)
    print("   MARS LANDER ULTIMATE - IA Learning Experience")
    print("="*60)
    print("\nL'IA va apprendre a atterrir. Observez sa progression!")
    print("\nControles:")
    print("  +/-     : Vitesse de simulation")
    print("  P       : Pause")
    print("  ESPACE  : Redemarrer")
    print("  T       : Afficher/Masquer la trace")
    print("  ESC     : Quitter")
    print("="*60 + "\n")

    # Boucle principale
    while running:
        dt = 1 / max(1, current_speed)
        clock.tick(current_speed)
        global_time += dt

        # Evenements
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_t:
                    show_trail = not show_trail
                elif event.key == pygame.K_SPACE:
                    v = j.je_relance_le_jeu(v)
                    ia.episodes_count += 1
                    last_destroyed = False
                    last_landed = False
                    terminal_time = None
                    trail.clear()
                    episode_start_time = time.time()
                elif event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS or event.key == pygame.K_EQUALS:
                    if speed_index < len(speed_levels) - 1:
                        speed_index += 1
                        current_speed = speed_levels[speed_index]
                elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                    if speed_index > 0:
                        speed_index -= 1
                        current_speed = speed_levels[speed_index]

        if not running:
            break

        # Mise a jour effets
        screen_shake.update(dt)
        celebration.update(dt)
        explosion.update(dt)
        progress_display.update(dt)

        # Logique du jeu
        if not paused and not v.detruit and not v.est_pose:
            # IA choisit une action
            etat = ia.recupere_etat(v, s)
            ia_action = ia.choisir_action(etat)

            if ia_active and ia_action:
                v.angle, v.puissance = ia_action

            # Mise a jour physique
            j.actualisation(v, a, s, ia)

            # Vent
            if data_module.vent_actif:
                wind.update()
                force = wind.get_force()
                v.h_speed += force[0]
                v.v_speed += force[1]

            # Enregistrer la trace
            speed = math.sqrt(v.h_speed**2 + v.v_speed**2)
            trail.add_point(v.x, fenY - v.y, speed)

            # Son
            if sound.initialized:
                sound.play_thrust(v.puissance)

        # Detection collision
        j.touche_mars(a, v, s)

        # Gestion fin d'episode
        vessel_screen_x = v.x / echelle
        vessel_screen_y = (fenY - v.y) / echelle

        if v.detruit and not last_destroyed:
            sound.play_explosion()
            sound.stop_all()
            terminal_time = time.time()
            episode_duration = time.time() - episode_start_time

            # Effets
            screen_shake.shake(15, 0.5)
            explosion.trigger(vessel_screen_x, vessel_screen_y)

            # Stats
            reward = ia.recompense if hasattr(ia, 'recompense') else 0
            progress_display.record_episode(reward, False, episode_duration)

        if v.est_pose and not last_landed:
            sound.play_success()
            sound.stop_all()
            terminal_time = time.time()
            episode_duration = time.time() - episode_start_time

            # Effets
            celebration.trigger(vessel_screen_x, vessel_screen_y)

            # Stats
            reward = ia.recompense if hasattr(ia, 'recompense') else 0
            progress_display.record_episode(reward, True, episode_duration)

        last_destroyed = v.detruit
        last_landed = v.est_pose

        # Apprentissage
        if ia_active and not paused:
            recompense = ia.recupere_recompense(a, v, s, j)
            ia.ajout_recompense_cumulative(recompense)
            next_etat = ia.recupere_etat(v, s)
            is_terminal = v.detruit or v.est_pose

            if etat and ia_action:
                ia.update_q_table(etat, ia_action, recompense, next_etat, is_terminal)
                ia.store_experience(etat, ia_action, recompense, next_etat, is_terminal)
                ia.train_on_batch()

            ia.decay_epsilon()

        # Relance auto
        is_terminal = v.detruit or v.est_pose
        if is_terminal and ia_active and not paused:
            if terminal_time and (time.time() - terminal_time) >= LANDING_DELAY:
                ia.end_episode()
                ia.episodes_count += 1
                v = j.je_relance_le_jeu(v)
                last_destroyed = False
                last_landed = False
                terminal_time = None
                trail.clear()
                episode_start_time = time.time()

        # =================================================================
        # RENDU
        # =================================================================

        # Offset de tremblement
        shake_offset = screen_shake.get_offset()

        # Fond noir avec etoiles
        screen.fill((5, 5, 15))
        stars.draw(screen, global_time)

        # Arriere-plan Mars (gradient)
        for y in range(screen_height // 2, screen_height):
            ratio = (y - screen_height // 2) / (screen_height // 2)
            r = int(30 + ratio * 60)
            g = int(15 + ratio * 30)
            b = int(10 + ratio * 20)
            pygame.draw.line(screen, (r, g, b), (0, y), (screen_width, y))

        # Trace de descente
        if show_trail:
            trail.draw(screen, echelle)

        # Surface de Mars
        a.dessiner_surface(s.mars_surface)

        # Vaisseau
        a.dessiner_vaisseau(v, j)

        # Trajectoire predictive
        if data_module.trajectoire_active:
            vent_force = wind.get_force() if data_module.vent_actif else (0, 0)
            a.dessiner_trajectoire(v, vent_force)

        # Indicateur vent
        if data_module.vent_actif:
            a.dessiner_indicateur_vent(wind.get_force())

        # Effets
        explosion.draw(screen)
        celebration.draw(screen)

        # HUD basique
        a.ecrire_info(v, ia, j)

        # Progression IA (le plus important!)
        progress_display.draw(screen, font_small, ia)

        # Indicateur de vitesse de simulation
        speed_text = font.render(f"x{current_speed} FPS", True, (150, 150, 150))
        screen.blit(speed_text, (10, screen_height - 25))

        # Indicateur de pause
        if paused:
            pause_surf = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
            pause_surf.fill((0, 0, 0, 150))
            screen.blit(pause_surf, (0, 0))

            pause_text = font_title.render("PAUSE", True, (255, 255, 255))
            screen.blit(pause_text, (screen_width//2 - pause_text.get_width()//2,
                                     screen_height//2 - pause_text.get_height()//2))

        pygame.display.flip()

    # Fin
    sound.stop_all()
    pygame.quit()

    # Stats finales
    print("\n" + "="*60)
    print("   SESSION TERMINEE")
    print("="*60)
    print(f"Episodes: {progress_display.total_episodes}")
    print(f"Reussites: {progress_display.total_successes}")
    print(f"Taux: {progress_display.get_success_rate():.1%}")
    print(f"Meilleure serie: {progress_display.best_streak}")
    print("="*60)


if __name__ == "__main__":
    main()
