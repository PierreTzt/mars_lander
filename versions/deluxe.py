"""
MARS LANDER DELUXE - L'Experience Complete

Menu interactif, modes de jeu, achievements, visualisations avancees,
heatmaps, replays, et bien plus encore!

Idee originale: Florent Lannois
Developpement: Claude AI

CARTE BLANCHE - AUCUNE LIMITE!
"""

import os
import time
import random
import math
import json
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from enum import Enum

import pygame
import numpy as np

# Imports locaux
from lander.data import *
from lander import data as data_module
from lander.vaisseau import Vaisseau
from lander.jeu import Jeu
from lander.affichage import Affichage
from lander.surface import Surface
from lander.ia_learning import IALearning
from lander.common import WindSystem, SoundManager
from lander.paths import chemin_sauvegarde


# =============================================================================
# CONSTANTES ET COULEURS
# =============================================================================

class Colors:
    """Palette de couleurs cyberpunk/sci-fi."""
    BLACK = (5, 5, 15)
    WHITE = (255, 255, 255)
    CYAN = (0, 255, 255)
    NEON_BLUE = (0, 150, 255)
    NEON_PINK = (255, 0, 150)
    NEON_GREEN = (0, 255, 100)
    NEON_ORANGE = (255, 150, 0)
    GOLD = (255, 215, 0)
    SILVER = (192, 192, 192)
    BRONZE = (205, 127, 50)
    RED = (255, 50, 50)
    DARK_BLUE = (10, 20, 40)
    PANEL_BG = (15, 25, 45, 230)


class GameState(Enum):
    """Etats du jeu."""
    MENU = "menu"
    PLAYING = "playing"
    PAUSED = "paused"
    GAME_OVER = "game_over"
    ACHIEVEMENTS = "achievements"
    STATS = "stats"
    SETTINGS = "settings"
    REPLAYS = "replays"


# =============================================================================
# SYSTEME D'ACHIEVEMENTS
# =============================================================================

@dataclass
class Achievement:
    """Un achievement deblocable."""
    id: str
    name: str
    description: str
    icon: str
    unlocked: bool = False
    unlock_time: float = 0
    rarity: str = "common"  # common, rare, epic, legendary


class AchievementSystem:
    """Gere les achievements du jeu."""

    def __init__(self):
        self.achievements: Dict[str, Achievement] = {}
        self.recently_unlocked: List[Achievement] = []
        self.notification_duration = 3.0
        self._init_achievements()
        self.load()

    def _init_achievements(self):
        """Initialise tous les achievements."""
        achievements_data = [
            ("first_landing", "Premier Pas", "Reussir votre premier atterrissage", "rocket", "common"),
            ("ten_landings", "Pilote Confirme", "Reussir 10 atterrissages", "star", "common"),
            ("hundred_landings", "As de l'Espace", "Reussir 100 atterrissages", "trophy", "rare"),
            ("streak_5", "Serie Gagnante", "5 atterrissages consecutifs", "fire", "common"),
            ("streak_10", "Inarretable", "10 atterrissages consecutifs", "fire", "rare"),
            ("streak_25", "Legende", "25 atterrissages consecutifs", "crown", "epic"),
            ("perfect_landing", "Atterrissage Parfait", "Atterrir avec plus de 80% de fuel", "diamond", "rare"),
            ("speed_demon", "Vitesse Lumiere", "Atterrir en moins de 5 secondes", "bolt", "epic"),
            ("fuel_saver", "Econome", "Atterrir avec 90% de fuel", "leaf", "legendary"),
            ("survivor", "Survivant", "Atteindre le niveau 10 en mode survie", "shield", "epic"),
            ("explorer", "Explorateur", "Jouer sur toutes les planetes", "globe", "rare"),
            ("night_owl", "Oiseau de Nuit", "10 atterrissages en mode nuit", "moon", "rare"),
            ("storm_rider", "Chasseur de Tempetes", "Atterrir pendant une tempete", "cloud", "rare"),
            ("collector", "Collectionneur", "Ramasser 50 power-ups", "gem", "common"),
            ("marathon", "Marathon", "Jouer pendant 1 heure", "clock", "common"),
            ("dedicated", "Dedie", "1000 episodes d'entrainement", "brain", "rare"),
            ("master", "Maitre", "Taux de reussite de 90%", "medal", "legendary"),
        ]

        for aid, name, desc, icon, rarity in achievements_data:
            self.achievements[aid] = Achievement(aid, name, desc, icon, rarity=rarity)

    def check_and_unlock(self, achievement_id: str) -> bool:
        """Verifie et debloque un achievement."""
        if achievement_id in self.achievements:
            ach = self.achievements[achievement_id]
            if not ach.unlocked:
                ach.unlocked = True
                ach.unlock_time = time.time()
                self.recently_unlocked.append(ach)
                self.save()
                return True
        return False

    def get_unlocked_count(self) -> int:
        return sum(1 for a in self.achievements.values() if a.unlocked)

    def get_total_count(self) -> int:
        return len(self.achievements)

    def save(self):
        """Sauvegarde les achievements."""
        data = {aid: ach.unlocked for aid, ach in self.achievements.items()}
        try:
            with open(chemin_sauvegarde('achievements.json'), 'w') as f:
                json.dump(data, f)
        except:
            pass

    def load(self):
        """Charge les achievements."""
        try:
            if os.path.exists(chemin_sauvegarde('achievements.json')):
                with open(chemin_sauvegarde('achievements.json'), 'r') as f:
                    data = json.load(f)
                for aid, unlocked in data.items():
                    if aid in self.achievements:
                        self.achievements[aid].unlocked = unlocked
        except:
            pass


# =============================================================================
# SYSTEME DE HEATMAP
# =============================================================================

class HeatmapSystem:
    """Visualise les zones de crash et d'atterrissage."""

    def __init__(self, width: int, height: int, cell_size: int = 20):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.grid_w = width // cell_size
        self.grid_h = height // cell_size
        self.crash_map = np.zeros((self.grid_h, self.grid_w))
        self.landing_map = np.zeros((self.grid_h, self.grid_w))

    def record_crash(self, x: float, y: float):
        gx = min(self.grid_w - 1, max(0, int(x / self.cell_size)))
        gy = min(self.grid_h - 1, max(0, int(y / self.cell_size)))
        self.crash_map[gy, gx] += 1

    def record_landing(self, x: float, y: float):
        gx = min(self.grid_w - 1, max(0, int(x / self.cell_size)))
        gy = min(self.grid_h - 1, max(0, int(y / self.cell_size)))
        self.landing_map[gy, gx] += 1

    def draw(self, screen, show_crashes: bool = True, show_landings: bool = True, alpha: int = 100):
        if show_crashes and self.crash_map.max() > 0:
            normalized = self.crash_map / self.crash_map.max()
            for gy in range(self.grid_h):
                for gx in range(self.grid_w):
                    if normalized[gy, gx] > 0.1:
                        intensity = int(normalized[gy, gx] * 255)
                        color = (intensity, 0, 0, min(alpha, intensity))
                        surf = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                        surf.fill(color)
                        screen.blit(surf, (gx * self.cell_size, gy * self.cell_size))

        if show_landings and self.landing_map.max() > 0:
            normalized = self.landing_map / self.landing_map.max()
            for gy in range(self.grid_h):
                for gx in range(self.grid_w):
                    if normalized[gy, gx] > 0.1:
                        intensity = int(normalized[gy, gx] * 255)
                        color = (0, intensity, 0, min(alpha, intensity))
                        surf = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                        surf.fill(color)
                        screen.blit(surf, (gx * self.cell_size, gy * self.cell_size))


# =============================================================================
# SYSTEME DE REPLAY
# =============================================================================

@dataclass
class ReplayFrame:
    """Une frame de replay."""
    x: float
    y: float
    angle: float
    puissance: int
    v_speed: float
    h_speed: float
    fuel: float


class ReplayRecorder:
    """Enregistre et rejoue les tentatives."""

    def __init__(self, max_replays: int = 10):
        self.current_recording: List[ReplayFrame] = []
        self.saved_replays: List[Tuple[List[ReplayFrame], bool, float]] = []  # (frames, success, score)
        self.max_replays = max_replays
        self.is_recording = False

    def start_recording(self):
        self.current_recording = []
        self.is_recording = True

    def record_frame(self, v):
        if self.is_recording:
            self.current_recording.append(ReplayFrame(
                x=v.x, y=v.y, angle=v.angle, puissance=v.puissance,
                v_speed=v.v_speed, h_speed=v.h_speed, fuel=v.fuel
            ))

    def end_recording(self, success: bool, score: float):
        if self.current_recording:
            self.saved_replays.append((self.current_recording.copy(), success, score))
            if len(self.saved_replays) > self.max_replays:
                # Garder les meilleurs
                self.saved_replays.sort(key=lambda x: x[2], reverse=True)
                self.saved_replays = self.saved_replays[:self.max_replays]
        self.is_recording = False

    def get_best_replays(self, n: int = 5) -> List[Tuple[List[ReplayFrame], bool, float]]:
        sorted_replays = sorted(self.saved_replays, key=lambda x: x[2], reverse=True)
        return sorted_replays[:n]


# =============================================================================
# PARTICULES AVANCEES
# =============================================================================

class ParticleSystem:
    """Systeme de particules avance."""

    def __init__(self, max_particles: int = 1000):
        self.particles = []
        self.max_particles = max_particles

    def emit(self, x: float, y: float, particle_type: str, count: int = 1):
        for _ in range(count):
            if len(self.particles) >= self.max_particles:
                break

            if particle_type == "explosion":
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(2, 8)
                self.particles.append({
                    'x': x, 'y': y,
                    'vx': math.cos(angle) * speed,
                    'vy': math.sin(angle) * speed - 2,
                    'color': random.choice([(255, 200, 50), (255, 150, 30), (255, 80, 20)]),
                    'size': random.uniform(3, 8),
                    'life': random.uniform(0.5, 1.5),
                    'gravity': 0.2,
                    'type': 'circle'
                })

            elif particle_type == "celebration":
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(3, 10)
                self.particles.append({
                    'x': x, 'y': y,
                    'vx': math.cos(angle) * speed,
                    'vy': math.sin(angle) * speed - 5,
                    'color': random.choice([Colors.GOLD, Colors.NEON_GREEN, Colors.CYAN, Colors.NEON_PINK]),
                    'size': random.uniform(2, 6),
                    'life': random.uniform(1, 3),
                    'gravity': 0.15,
                    'type': 'star'
                })

            elif particle_type == "thrust":
                angle = random.uniform(-0.3, 0.3)
                speed = random.uniform(2, 5)
                self.particles.append({
                    'x': x, 'y': y,
                    'vx': math.sin(angle) * speed * 0.3,
                    'vy': speed,
                    'color': random.choice([(255, 200, 100), (255, 150, 50)]),
                    'size': random.uniform(2, 4),
                    'life': random.uniform(0.2, 0.5),
                    'gravity': -0.1,
                    'type': 'circle'
                })

            elif particle_type == "dust":
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(0.5, 2)
                self.particles.append({
                    'x': x, 'y': y,
                    'vx': math.cos(angle) * speed,
                    'vy': -abs(math.sin(angle) * speed),
                    'color': (180, 140, 100),
                    'size': random.uniform(2, 5),
                    'life': random.uniform(1, 2),
                    'gravity': 0.05,
                    'type': 'circle'
                })

    def update(self, dt: float):
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

            if p['type'] == 'circle':
                surf = pygame.Surface((int(p['size']*2), int(p['size']*2)), pygame.SRCALPHA)
                pygame.draw.circle(surf, color, (int(p['size']), int(p['size'])), int(p['size']))
                screen.blit(surf, (int(p['x'] - p['size']), int(p['y'] - p['size'])))

            elif p['type'] == 'star':
                self._draw_star(screen, p['x'], p['y'], p['size'], color)

    def _draw_star(self, screen, x, y, size, color):
        points = []
        for i in range(5):
            angle = -math.pi/2 + i * 2 * math.pi / 5
            points.append((x + size * math.cos(angle), y + size * math.sin(angle)))
            angle += math.pi / 5
            points.append((x + size * 0.4 * math.cos(angle), y + size * 0.4 * math.sin(angle)))

        surf = pygame.Surface((int(size*3), int(size*3)), pygame.SRCALPHA)
        adjusted_points = [(px - x + size*1.5, py - y + size*1.5) for px, py in points]
        pygame.draw.polygon(surf, color, adjusted_points)
        screen.blit(surf, (int(x - size*1.5), int(y - size*1.5)))


# =============================================================================
# UI COMPONENTS
# =============================================================================

class Button:
    """Bouton interactif."""

    def __init__(self, x: int, y: int, width: int, height: int, text: str,
                 color: Tuple = Colors.NEON_BLUE, hover_color: Tuple = Colors.CYAN):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.hovered = False
        self.pulse = 0

    def update(self, mouse_pos: Tuple[int, int], dt: float):
        self.hovered = self.rect.collidepoint(mouse_pos)
        self.pulse = (self.pulse + dt * 5) % (2 * math.pi)

    def draw(self, screen, font):
        color = self.hover_color if self.hovered else self.color

        # Effet de lueur
        if self.hovered:
            glow_size = 5 + int(3 * math.sin(self.pulse))
            glow_surf = pygame.Surface((self.rect.width + glow_size*2, self.rect.height + glow_size*2), pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, (*color, 50), (0, 0, glow_surf.get_width(), glow_surf.get_height()), border_radius=10)
            screen.blit(glow_surf, (self.rect.x - glow_size, self.rect.y - glow_size))

        # Fond
        pygame.draw.rect(screen, Colors.DARK_BLUE, self.rect, border_radius=8)
        pygame.draw.rect(screen, color, self.rect, 2, border_radius=8)

        # Texte
        text_surf = font.render(self.text, True, color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def is_clicked(self, mouse_pos: Tuple[int, int], mouse_pressed: bool) -> bool:
        return self.rect.collidepoint(mouse_pos) and mouse_pressed


class ProgressBar:
    """Barre de progression animee."""

    def __init__(self, x: int, y: int, width: int, height: int,
                 color: Tuple = Colors.NEON_GREEN):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.value = 0  # 0-1
        self.target_value = 0
        self.pulse = 0

    def set_value(self, value: float):
        self.target_value = max(0, min(1, value))

    def update(self, dt: float):
        # Animation fluide
        self.value += (self.target_value - self.value) * dt * 5
        self.pulse = (self.pulse + dt * 3) % (2 * math.pi)

    def draw(self, screen):
        # Fond
        pygame.draw.rect(screen, (30, 30, 50), self.rect, border_radius=4)

        # Remplissage
        fill_width = int(self.rect.width * self.value)
        if fill_width > 0:

            # Gradient effect
            for i in range(fill_width):
                ratio = i / max(1, fill_width)
                brightness = 0.7 + 0.3 * math.sin(self.pulse + ratio * 3)
                color = tuple(int(c * brightness) for c in self.color)
                pygame.draw.line(screen, color,
                               (self.rect.x + i, self.rect.y),
                               (self.rect.x + i, self.rect.y + self.rect.height - 1))

        # Bordure
        pygame.draw.rect(screen, self.color, self.rect, 1, border_radius=4)


# =============================================================================
# MENU PRINCIPAL
# =============================================================================

class MainMenu:
    """Menu principal du jeu."""

    def __init__(self, screen_width: int, screen_height: int):
        self.width = screen_width
        self.height = screen_height

        # Boutons
        btn_width = 250
        btn_height = 50
        btn_x = screen_width // 2 - btn_width // 2
        start_y = 250

        self.buttons = {
            'play': Button(btn_x, start_y, btn_width, btn_height, "JOUER", Colors.NEON_GREEN),
            'training': Button(btn_x, start_y + 70, btn_width, btn_height, "ENTRAINER L'IA", Colors.NEON_BLUE),
            'achievements': Button(btn_x, start_y + 140, btn_width, btn_height, "ACHIEVEMENTS", Colors.GOLD),
            'stats': Button(btn_x, start_y + 210, btn_width, btn_height, "STATISTIQUES", Colors.NEON_PINK),
            'quit': Button(btn_x, start_y + 280, btn_width, btn_height, "QUITTER", Colors.RED),
        }

        # Animation
        self.title_pulse = 0
        self.stars = self._generate_stars()

    def _generate_stars(self):
        stars = []
        for _ in range(100):
            stars.append({
                'x': random.randint(0, self.width),
                'y': random.randint(0, self.height),
                'size': random.uniform(0.5, 2),
                'speed': random.uniform(0.5, 2)
            })
        return stars

    def update(self, dt: float, mouse_pos: Tuple[int, int]):
        self.title_pulse = (self.title_pulse + dt * 2) % (2 * math.pi)

        for btn in self.buttons.values():
            btn.update(mouse_pos, dt)

        # Animation des etoiles
        for star in self.stars:
            star['y'] += star['speed']
            if star['y'] > self.height:
                star['y'] = 0
                star['x'] = random.randint(0, self.width)

    def draw(self, screen, fonts: Dict[str, pygame.font.Font]):
        # Fond
        screen.fill(Colors.BLACK)

        # Etoiles
        for star in self.stars:
            brightness = int(150 + 50 * random.random())
            pygame.draw.circle(screen, (brightness, brightness, brightness),
                             (int(star['x']), int(star['y'])), int(star['size']))

        # Planete Mars en arriere-plan
        pygame.draw.circle(screen, (80, 30, 20), (self.width - 100, self.height - 80), 120)
        pygame.draw.circle(screen, (100, 40, 25), (self.width - 100, self.height - 80), 115)

        # Titre avec effet de lueur
        title_text = "MARS LANDER"

        # Lueur
        for offset in range(3, 0, -1):
            alpha = 50 // offset
            glow = fonts['title'].render(title_text, True, (*Colors.NEON_BLUE[:3], alpha))
            screen.blit(glow, (self.width//2 - glow.get_width()//2 - offset,
                              60 - offset))
            screen.blit(glow, (self.width//2 - glow.get_width()//2 + offset,
                              60 + offset))

        title_surf = fonts['title'].render(title_text, True, Colors.WHITE)
        screen.blit(title_surf, (self.width//2 - title_surf.get_width()//2, 60))

        # Sous-titre
        subtitle = fonts['medium'].render("DELUXE EDITION", True, Colors.GOLD)
        screen.blit(subtitle, (self.width//2 - subtitle.get_width()//2, 130))

        # Credits
        credits = fonts['small'].render("Idee: Florent Lannois | Dev: Claude AI", True, (100, 100, 100))
        screen.blit(credits, (self.width//2 - credits.get_width()//2, 180))

        # Boutons
        for btn in self.buttons.values():
            btn.draw(screen, fonts['medium'])

    def handle_click(self, mouse_pos: Tuple[int, int]) -> Optional[str]:
        for name, btn in self.buttons.items():
            if btn.rect.collidepoint(mouse_pos):
                return name
        return None


# =============================================================================
# AFFICHAGE DES STATS
# =============================================================================

class StatsDisplay:
    """Affichage detaille des statistiques."""

    def __init__(self):
        self.total_episodes = 0
        self.total_landings = 0
        self.total_crashes = 0
        self.total_time = 0
        self.best_streak = 0
        self.rewards_history = deque(maxlen=1000)
        self.session_start = time.time()

    def update_stats(self, ia, success: bool, reward: float):
        self.total_episodes += 1
        if success:
            self.total_landings += 1
        else:
            self.total_crashes += 1
        self.rewards_history.append(reward)

    def draw(self, screen, fonts, width, height):
        # Fond
        screen.fill(Colors.BLACK)

        # Titre
        title = fonts['large'].render("STATISTIQUES", True, Colors.CYAN)
        screen.blit(title, (width//2 - title.get_width()//2, 30))

        y = 100
        stats = [
            (f"Episodes totaux: {self.total_episodes}", Colors.WHITE),
            (f"Atterrissages reussis: {self.total_landings}", Colors.NEON_GREEN),
            (f"Crashes: {self.total_crashes}", Colors.RED),
            (f"Taux de reussite: {self.total_landings/max(1,self.total_episodes):.1%}", Colors.GOLD),
            (f"Meilleure serie: {self.best_streak}", Colors.NEON_PINK),
        ]

        for text, color in stats:
            surf = fonts['medium'].render(text, True, color)
            screen.blit(surf, (50, y))
            y += 40

        # Graphique
        if len(self.rewards_history) > 1:
            self._draw_graph(screen, 50, 350, width - 100, 200, fonts)

        # Instructions
        inst = fonts['small'].render("Appuyez sur ECHAP pour revenir", True, (100, 100, 100))
        screen.blit(inst, (width//2 - inst.get_width()//2, height - 40))

    def _draw_graph(self, screen, x, y, width, height, fonts):
        # Titre du graphique
        title = fonts['small'].render("Evolution des recompenses", True, Colors.WHITE)
        screen.blit(title, (x, y - 25))

        # Fond
        pygame.draw.rect(screen, (20, 25, 40), (x, y, width, height))
        pygame.draw.rect(screen, Colors.NEON_BLUE, (x, y, width, height), 1)

        data = list(self.rewards_history)
        if len(data) < 2:
            return

        min_val = min(data)
        max_val = max(data)
        if max_val == min_val:
            max_val = min_val + 1

        points = []
        for i, val in enumerate(data):
            px = x + (i / (len(data) - 1)) * width
            py = y + height - ((val - min_val) / (max_val - min_val)) * height
            points.append((px, py))

        # Ligne zero
        if min_val < 0 < max_val:
            zero_y = y + height - ((0 - min_val) / (max_val - min_val)) * height
            pygame.draw.line(screen, (60, 60, 60), (x, zero_y), (x + width, zero_y))

        # Courbe
        if len(points) >= 2:
            pygame.draw.lines(screen, Colors.NEON_GREEN, False, points, 2)


# =============================================================================
# AFFICHAGE DES ACHIEVEMENTS
# =============================================================================

class AchievementsDisplay:
    """Affichage des achievements."""

    def __init__(self, achievement_system: AchievementSystem):
        self.system = achievement_system
        self.scroll_offset = 0

    def draw(self, screen, fonts, width, height):
        screen.fill(Colors.BLACK)

        # Titre
        title = fonts['large'].render("ACHIEVEMENTS", True, Colors.GOLD)
        screen.blit(title, (width//2 - title.get_width()//2, 20))

        # Progression
        unlocked = self.system.get_unlocked_count()
        total = self.system.get_total_count()
        progress = fonts['medium'].render(f"{unlocked} / {total} debloque(s)", True, Colors.WHITE)
        screen.blit(progress, (width//2 - progress.get_width()//2, 70))

        # Liste des achievements
        y = 120
        for ach in self.system.achievements.values():
            self._draw_achievement(screen, fonts, 50, y, width - 100, ach)
            y += 60

        # Instructions
        inst = fonts['small'].render("Appuyez sur ECHAP pour revenir", True, (100, 100, 100))
        screen.blit(inst, (width//2 - inst.get_width()//2, height - 30))

    def _draw_achievement(self, screen, fonts, x, y, width, ach: Achievement):
        height = 50

        # Couleurs selon rarete
        rarity_colors = {
            'common': (150, 150, 150),
            'rare': (0, 150, 255),
            'epic': (200, 0, 255),
            'legendary': (255, 200, 0)
        }
        border_color = rarity_colors.get(ach.rarity, Colors.WHITE)

        # Fond
        bg_color = (30, 35, 50) if ach.unlocked else (20, 20, 25)
        pygame.draw.rect(screen, bg_color, (x, y, width, height), border_radius=8)
        pygame.draw.rect(screen, border_color if ach.unlocked else (50, 50, 50),
                        (x, y, width, height), 2, border_radius=8)

        # Icone (placeholder)
        icon_color = border_color if ach.unlocked else (60, 60, 60)
        pygame.draw.circle(screen, icon_color, (x + 25, y + height//2), 15)

        # Nom
        name_color = Colors.WHITE if ach.unlocked else (80, 80, 80)
        name = fonts['medium'].render(ach.name, True, name_color)
        screen.blit(name, (x + 50, y + 5))

        # Description
        desc_color = (150, 150, 150) if ach.unlocked else (60, 60, 60)
        desc = fonts['small'].render(ach.description, True, desc_color)
        screen.blit(desc, (x + 50, y + 28))

        # Rarete
        rarity_text = fonts['small'].render(ach.rarity.upper(), True, border_color)
        screen.blit(rarity_text, (x + width - rarity_text.get_width() - 10, y + height//2 - 8))


# =============================================================================
# JEU PRINCIPAL
# =============================================================================

class Game:
    """Classe principale du jeu."""

    def __init__(self):
        pygame.init()
        pygame.mixer.init()

        # Ecran
        self.width = fenX // echelle
        self.height = fenY // echelle
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("MARS LANDER DELUXE")

        # Polices
        self.fonts = {
            'small': pygame.font.Font(None, 18),
            'medium': pygame.font.Font(None, 24),
            'large': pygame.font.Font(None, 36),
            'title': pygame.font.Font(None, 72)
        }

        # Etat
        self.state = GameState.MENU
        self.running = True
        self.clock = pygame.time.Clock()

        # Composants du jeu
        self.scenar = scenario0
        self.vessel = None
        self.surface = None
        self.affichage = None
        self.jeu = None
        self.ia = None

        # Systemes
        self.achievements = AchievementSystem()
        self.particles = ParticleSystem()
        self.heatmap = HeatmapSystem(self.width, self.height)
        self.replay_recorder = ReplayRecorder()
        self.stats = StatsDisplay()

        # UI
        self.menu = MainMenu(self.width, self.height)
        self.achievements_display = AchievementsDisplay(self.achievements)

        # Variables de jeu
        self.speed_levels = [30, 60, 120, 500, 1000, 5000]
        self.speed_index = 2
        self.current_streak = 0
        self.show_heatmap = False

        # Sons
        self.sound = SoundManager()
        self.wind = WindSystem()

    def init_game(self, training_mode: bool = True):
        """Initialise une nouvelle partie."""
        self.vessel = Vaisseau()
        self.vessel.init_vaisseau(self.scenar['vaisseau'])
        self.surface = Surface(self.scenar['surface_mars'])
        self.affichage = Affichage()
        self.jeu = Jeu(self.scenar)

        zone = self.surface.calcul_zone_atterissage(self.scenar)
        self.affichage.init_terrain(self.scenar['surface_mars'], zone)
        self.affichage.set_landing_zone(zone)

        toutes_actions = self.jeu.toutes_actions_possibles(self.vessel)

        # Mode IA ou manuel
        ia_mode = training_mode
        self.ia = IALearning(self.scenar, toutes_actions, alpha, gamma, epsilon, epsilon_decay, ia_mode)

        self.replay_recorder.start_recording()
        self.episode_start_time = time.time()
        self.last_destroyed = False
        self.last_landed = False
        self.terminal_time = None

    def run(self):
        """Boucle principale."""
        while self.running:
            dt = self.clock.tick(60) / 1000.0

            if self.state == GameState.MENU:
                self._handle_menu(dt)
            elif self.state == GameState.PLAYING:
                self._handle_game(dt)
            elif self.state == GameState.ACHIEVEMENTS:
                self._handle_achievements(dt)
            elif self.state == GameState.STATS:
                self._handle_stats(dt)

        pygame.quit()

    def _handle_menu(self, dt: float):
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                action = self.menu.handle_click(mouse_pos)
                if action == 'play':
                    self.init_game(training_mode=False)
                    self.state = GameState.PLAYING
                elif action == 'training':
                    self.init_game(training_mode=True)
                    self.state = GameState.PLAYING
                elif action == 'achievements':
                    self.state = GameState.ACHIEVEMENTS
                elif action == 'stats':
                    self.state = GameState.STATS
                elif action == 'quit':
                    self.running = False

        self.menu.update(dt, mouse_pos)
        self.menu.draw(self.screen, self.fonts)
        pygame.display.flip()

    def _handle_game(self, dt: float):
        current_speed = self.speed_levels[self.speed_index]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.state = GameState.MENU
                elif event.key == pygame.K_SPACE:
                    self._restart_episode()
                elif event.key == pygame.K_h:
                    self.show_heatmap = not self.show_heatmap
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    if self.speed_index < len(self.speed_levels) - 1:
                        self.speed_index += 1
                elif event.key == pygame.K_MINUS:
                    if self.speed_index > 0:
                        self.speed_index -= 1

        # Simulation
        self.clock.tick(current_speed)

        # Initialisation des variables pour l'apprentissage
        etat = None
        ia_action = None

        if not self.vessel.detruit and not self.vessel.est_pose:
            # IA action
            etat = self.ia.recupere_etat(self.vessel, self.surface)
            ia_action = self.ia.choisir_action(etat)

            if ia_active and ia_action:
                self.vessel.angle, self.vessel.puissance = ia_action

            # Physique
            self.jeu.actualisation(self.vessel, self.affichage, self.surface, self.ia)

            # Vent
            if data_module.vent_actif:
                self.wind.update()
                force = self.wind.get_force()
                self.vessel.h_speed += force[0]
                self.vessel.v_speed += force[1]

            # Particules de propulsion
            if self.vessel.puissance > 0:
                vx = self.vessel.x / echelle
                vy = (fenY - self.vessel.y) / echelle
                self.particles.emit(vx, vy + 15, "thrust", self.vessel.puissance)

            # Enregistrement
            self.replay_recorder.record_frame(self.vessel)

        # Detection collision
        self.jeu.touche_mars(self.affichage, self.vessel, self.surface)

        vx = self.vessel.x / echelle
        vy = (fenY - self.vessel.y) / echelle

        # Fin episode
        if self.vessel.detruit and not self.last_destroyed:
            self.sound.play_explosion()
            self.terminal_time = time.time()
            self.particles.emit(vx, vy, "explosion", 80)
            self.heatmap.record_crash(vx, vy)
            self.current_streak = 0
            reward = self.ia.recompense if hasattr(self.ia, 'recompense') else 0
            self.stats.update_stats(self.ia, False, reward)
            self.replay_recorder.end_recording(False, reward)

        if self.vessel.est_pose and not self.last_landed:
            self.sound.play_success()
            self.terminal_time = time.time()
            self.particles.emit(vx, vy, "celebration", 100)
            self.particles.emit(vx, vy + 10, "dust", 30)
            self.heatmap.record_landing(vx, vy)
            self.current_streak += 1
            self.stats.best_streak = max(self.stats.best_streak, self.current_streak)

            reward = self.ia.recompense if hasattr(self.ia, 'recompense') else 0
            self.stats.update_stats(self.ia, True, reward)
            self.replay_recorder.end_recording(True, reward)

            # Achievements
            self.achievements.check_and_unlock("first_landing")
            if self.stats.total_landings >= 10:
                self.achievements.check_and_unlock("ten_landings")
            if self.stats.total_landings >= 100:
                self.achievements.check_and_unlock("hundred_landings")
            if self.current_streak >= 5:
                self.achievements.check_and_unlock("streak_5")
            if self.current_streak >= 10:
                self.achievements.check_and_unlock("streak_10")
            if self.current_streak >= 25:
                self.achievements.check_and_unlock("streak_25")

        self.last_destroyed = self.vessel.detruit
        self.last_landed = self.vessel.est_pose

        # Apprentissage
        if ia_active:
            recompense = self.ia.recupere_recompense(self.affichage, self.vessel, self.surface, self.jeu)
            self.ia.ajout_recompense_cumulative(recompense)
            next_etat = self.ia.recupere_etat(self.vessel, self.surface)
            is_terminal = self.vessel.detruit or self.vessel.est_pose

            if etat and ia_action:
                self.ia.update_q_table(etat, ia_action, recompense, next_etat, is_terminal)
                self.ia.store_experience(etat, ia_action, recompense, next_etat, is_terminal)
                self.ia.train_on_batch()
            self.ia.decay_epsilon()

        # Auto restart
        if (self.vessel.detruit or self.vessel.est_pose) and ia_active:
            if self.terminal_time and (time.time() - self.terminal_time) >= 1.5:
                self._restart_episode()

        # Mise a jour particules
        self.particles.update(dt)

        # Rendu
        self._render_game()

    def _restart_episode(self):
        self.ia.end_episode()
        self.ia.episodes_count += 1
        self.vessel = self.jeu.je_relance_le_jeu(self.vessel)
        self.last_destroyed = False
        self.last_landed = False
        self.terminal_time = None
        self.replay_recorder.start_recording()
        self.episode_start_time = time.time()

    def _render_game(self):
        # Fond
        self.screen.fill(Colors.BLACK)

        # Heatmap
        if self.show_heatmap:
            self.heatmap.draw(self.screen)

        # Jeu
        self.affichage.effacer_tout()
        self.affichage.dessiner_surface(self.surface.mars_surface)
        self.affichage.dessiner_vaisseau(self.vessel, self.jeu)

        # Particules
        self.particles.draw(self.screen)

        # HUD
        self.affichage.ecrire_info(self.vessel, self.ia, self.jeu)

        # Stats en haut a droite
        self._draw_stats_panel()

        # Vitesse
        speed_text = self.fonts['small'].render(f"x{self.speed_levels[self.speed_index]} FPS | H: Heatmap",
                                                True, (100, 100, 100))
        self.screen.blit(speed_text, (10, self.height - 20))

        pygame.display.flip()

    def _draw_stats_panel(self):
        panel_width = 200
        panel_height = 120
        x = self.width - panel_width - 10
        y = 10

        # Fond
        surf = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        surf.fill(Colors.PANEL_BG)
        self.screen.blit(surf, (x, y))
        pygame.draw.rect(self.screen, Colors.NEON_BLUE, (x, y, panel_width, panel_height), 1)

        # Stats
        stats_y = y + 10
        texts = [
            (f"Episode: {self.ia.episodes_count}", Colors.WHITE),
            (f"Reussites: {self.stats.total_landings}", Colors.NEON_GREEN),
            (f"Serie: {self.current_streak}", Colors.GOLD),
            (f"Taux: {self.stats.total_landings/max(1,self.stats.total_episodes):.0%}", Colors.CYAN),
            (f"Epsilon: {self.ia.epsilon:.2f}", Colors.NEON_PINK),
        ]

        for text, color in texts:
            surf = self.fonts['small'].render(text, True, color)
            self.screen.blit(surf, (x + 10, stats_y))
            stats_y += 20

    def _handle_achievements(self, dt: float):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.state = GameState.MENU

        self.achievements_display.draw(self.screen, self.fonts, self.width, self.height)
        pygame.display.flip()

    def _handle_stats(self, dt: float):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.state = GameState.MENU

        self.stats.draw(self.screen, self.fonts, self.width, self.height)
        pygame.display.flip()


# =============================================================================
# POINT D'ENTREE
# =============================================================================

def main():
    print("\n" + "="*60)
    print("   MARS LANDER DELUXE")
    print("   L'Experience Complete")
    print("="*60)
    print("\nIdee originale: Florent Lannois")
    print("Developpement: Claude AI")
    print("="*60 + "\n")

    game = Game()
    game.run()


if __name__ == "__main__":
    main()
