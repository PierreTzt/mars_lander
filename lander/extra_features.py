"""
Fonctionnalités supplémentaires pour Mars Lander.

Ce module contient:
- Éditeur de niveaux
- Terrain destructible
- Mode multijoueur local
- Export GIF/Vidéo
- Dashboard statistiques
"""

import pygame
import math
import os
import pickle
import time
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
from collections import deque

from .paths import chemin_sauvegarde


# =============================================================================
# 13. TERRAIN DESTRUCTIBLE
# =============================================================================

class DestructibleTerrain:
    """
    Gère le terrain destructible.
    Les crashes créent des cratères.
    """

    def __init__(self, surface_points: List[Tuple[int, int]], echelle: int = 10):
        self.original_points = [p for p in surface_points]
        self.points = [list(p) for p in surface_points]
        self.echelle = echelle
        self.craters: List[Tuple[float, float, float]] = []  # (x, y, radius)
        self.enabled = True

    def create_crater(self, x: float, y: float, impact_speed: float = 50) -> None:
        """
        Crée un cratère à l'impact.

        Args:
            x, y: Position de l'impact
            impact_speed: Vitesse d'impact (détermine la taille du cratère)
        """
        if not self.enabled:
            return

        # Taille du cratère basée sur la vitesse d'impact
        radius = min(100, max(20, impact_speed * 0.5))
        self.craters.append((x, y, radius))

        # Modifier les points du terrain
        for i, (px, py) in enumerate(self.points):
            dist = math.sqrt((px - x) ** 2 + (py - y) ** 2)
            if dist < radius:
                # Creuser le terrain
                depth = (radius - dist) * 0.3
                self.points[i][1] = py + depth  # Baisser le point

    def get_modified_surface(self) -> List[Tuple[int, int]]:
        """Retourne la surface modifiée."""
        return [(int(p[0]), int(p[1])) for p in self.points]

    def reset(self) -> None:
        """Réinitialise le terrain."""
        self.points = [list(p) for p in self.original_points]
        self.craters = []

    def draw_craters(self, screen: pygame.Surface) -> None:
        """Dessine les cratères."""
        for x, y, radius in self.craters:
            # Ombre du cratère
            crater_color = (80, 40, 20)
            pygame.draw.circle(screen, crater_color,
                             (int(x / self.echelle), int(y / self.echelle)),
                             int(radius / self.echelle))

            # Bord du cratère
            rim_color = (120, 60, 30)
            pygame.draw.circle(screen, rim_color,
                             (int(x / self.echelle), int(y / self.echelle)),
                             int(radius / self.echelle), 2)


# =============================================================================
# 14. MODE MULTIJOUEUR LOCAL
# =============================================================================

@dataclass
class Player:
    """Représente un joueur."""
    id: int
    name: str
    color: Tuple[int, int, int]
    vessel: Any = None  # Référence au vaisseau
    score: int = 0
    landings: int = 0
    crashes: int = 0
    controls: Dict[str, int] = field(default_factory=dict)


class MultiplayerMode:
    """
    Mode multijoueur local pour 2 joueurs.
    Course à l'atterrissage.
    """

    def __init__(self):
        self.enabled = False
        self.players: List[Player] = []
        self.winner: Optional[Player] = None
        self.round_in_progress = False
        self.rounds_played = 0
        self.max_rounds = 5

        # Initialiser les contrôles par défaut
        self._init_players()

    def _init_players(self) -> None:
        """Initialise les joueurs avec leurs contrôles."""
        # Joueur 1 (WASD + Space)
        p1_controls = {
            'left': pygame.K_a,
            'right': pygame.K_d,
            'thrust_up': pygame.K_w,
            'thrust_down': pygame.K_s,
        }

        # Joueur 2 (Flèches)
        p2_controls = {
            'left': pygame.K_LEFT,
            'right': pygame.K_RIGHT,
            'thrust_up': pygame.K_UP,
            'thrust_down': pygame.K_DOWN,
        }

        self.players = [
            Player(id=1, name="Joueur 1", color=(0, 150, 255), controls=p1_controls),
            Player(id=2, name="Joueur 2", color=(255, 150, 0), controls=p2_controls),
        ]

    def start_round(self) -> None:
        """Démarre un nouveau round."""
        self.round_in_progress = True
        self.winner = None

    def on_landing(self, player_id: int, fuel_remaining: float) -> int:
        """Appelé quand un joueur atterrit."""
        player = self.players[player_id - 1]
        player.landings += 1

        # Points: 1000 + bonus fuel
        points = 1000 + int(fuel_remaining)
        player.score += points

        if self.winner is None:
            self.winner = player
            points += 500  # Bonus pour le premier

        self.round_in_progress = False
        self.rounds_played += 1

        return points

    def on_crash(self, player_id: int) -> None:
        """Appelé quand un joueur crash."""
        player = self.players[player_id - 1]
        player.crashes += 1

    def handle_input(self, keys, player_id: int, vessel) -> None:
        """Gère les entrées d'un joueur."""
        if not self.enabled or player_id > len(self.players):
            return

        player = self.players[player_id - 1]
        controls = player.controls

        # Rotation
        if keys[controls['left']]:
            vessel.angle += 15
        if keys[controls['right']]:
            vessel.angle -= 15

        # Limiter l'angle
        vessel.angle = max(-90, min(90, vessel.angle))

        # Puissance
        if keys[controls['thrust_up']]:
            vessel.puissance = min(4, vessel.puissance + 1)
        if keys[controls['thrust_down']]:
            vessel.puissance = max(0, vessel.puissance - 1)

    def get_leader(self) -> Optional[Player]:
        """Retourne le joueur en tête."""
        if not self.players:
            return None
        return max(self.players, key=lambda p: p.score)

    def is_game_over(self) -> bool:
        """Vérifie si la partie est terminée."""
        return self.rounds_played >= self.max_rounds

    def get_final_winner(self) -> Optional[Player]:
        """Retourne le gagnant final."""
        if not self.is_game_over():
            return None
        return self.get_leader()

    def reset(self) -> None:
        """Réinitialise la partie."""
        for player in self.players:
            player.score = 0
            player.landings = 0
            player.crashes = 0
        self.rounds_played = 0
        self.winner = None
        self.round_in_progress = False

    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        """Affiche les scores des joueurs."""
        if not self.enabled:
            return

        y_offset = 10

        for player in self.players:
            # Fond coloré
            bg_surface = pygame.Surface((200, 50), pygame.SRCALPHA)
            bg_surface.fill((*player.color, 100))
            screen.blit(bg_surface, (10, y_offset))

            # Nom et score
            name_text = font.render(f"{player.name}", True, player.color)
            score_text = font.render(f"Score: {player.score}", True, (255, 255, 255))
            stats_text = font.render(f"L:{player.landings} C:{player.crashes}",
                                    True, (200, 200, 200))

            screen.blit(name_text, (15, y_offset + 2))
            screen.blit(score_text, (15, y_offset + 18))
            screen.blit(stats_text, (15, y_offset + 34))

            y_offset += 60

        # Round actuel
        round_text = font.render(f"Round {self.rounds_played + 1}/{self.max_rounds}",
                                True, (255, 255, 0))
        screen.blit(round_text, (10, y_offset))


# =============================================================================
# 15. EXPORT GIF/VIDÉO
# =============================================================================

class ScreenRecorder:
    """
    Enregistre les frames pour export en GIF ou vidéo.
    Utilise PIL si disponible, sinon sauvegarde en images.
    """

    def __init__(self, max_frames: int = 500, fps: int = 30):
        self.max_frames = max_frames
        self.fps = fps
        self.frames: List[pygame.Surface] = []
        self.recording = False
        self.enabled = True

        # Vérifier si PIL est disponible
        try:
            from PIL import Image  # noqa: F401 (test de disponibilité)
            self.pil_available = True
        except ImportError:
            self.pil_available = False
            print("Note: PIL non installé, export GIF désactivé. Installer avec: pip install Pillow")

    def start_recording(self) -> None:
        """Démarre l'enregistrement."""
        self.frames = []
        self.recording = True

    def stop_recording(self) -> None:
        """Arrête l'enregistrement."""
        self.recording = False

    def capture_frame(self, screen: pygame.Surface) -> None:
        """Capture une frame."""
        if not self.recording or not self.enabled:
            return

        if len(self.frames) < self.max_frames:
            # Copier la surface
            frame = screen.copy()
            self.frames.append(frame)

    def export_gif(self, filename: str = chemin_sauvegarde("landing.gif"), scale: float = 0.5) -> bool:
        """
        Exporte l'enregistrement en GIF.

        Args:
            filename: Nom du fichier de sortie
            scale: Facteur de redimensionnement (0.5 = moitié)

        Returns:
            True si succès
        """
        if not self.pil_available:
            print("PIL non disponible pour l'export GIF")
            return False

        if not self.frames:
            print("Aucune frame à exporter")
            return False

        from PIL import Image

        try:
            pil_frames = []

            for frame in self.frames:
                # Convertir pygame surface en string de pixels
                frame_str = pygame.image.tostring(frame, 'RGB')
                size = frame.get_size()

                # Créer image PIL
                img = Image.frombytes('RGB', size, frame_str)

                # Redimensionner si nécessaire
                if scale != 1.0:
                    new_size = (int(size[0] * scale), int(size[1] * scale))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)

                pil_frames.append(img)

            # Sauvegarder le GIF
            duration = int(1000 / self.fps)  # Durée de chaque frame en ms
            pil_frames[0].save(
                filename,
                save_all=True,
                append_images=pil_frames[1:],
                duration=duration,
                loop=0
            )

            print(f"GIF exporté: {filename} ({len(pil_frames)} frames)")
            return True

        except Exception as e:
            print(f"Erreur lors de l'export GIF: {e}")
            return False

    def export_images(self, folder: str = chemin_sauvegarde("frames")) -> bool:
        """Exporte les frames en images PNG."""
        if not self.frames:
            return False

        # Créer le dossier
        os.makedirs(folder, exist_ok=True)

        for i, frame in enumerate(self.frames):
            filename = os.path.join(folder, f"frame_{i:04d}.png")
            pygame.image.save(frame, filename)

        print(f"Images exportées dans {folder}/ ({len(self.frames)} frames)")
        return True

    def get_frame_count(self) -> int:
        """Retourne le nombre de frames enregistrées."""
        return len(self.frames)


# =============================================================================
# 16. ÉDITEUR DE NIVEAUX
# =============================================================================

class LevelEditor:
    """
    Éditeur de niveaux interactif.
    Permet de créer des terrains personnalisés avec la souris.
    """

    def __init__(self, screen_width: int = 1000, screen_height: int = 600, echelle: int = 10):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.echelle = echelle
        self.enabled = False
        self.active = False

        # Points du terrain
        self.points: List[Tuple[int, int]] = []
        self.landing_zone_start: Optional[int] = None
        self.landing_zone_end: Optional[int] = None

        # État de l'éditeur
        self.mode = 'draw'  # 'draw', 'landing_zone', 'vessel'
        self.vessel_position: Optional[Tuple[int, int]] = None

        # Terrain sauvegardé
        self.saved_levels: List[Dict] = []

    def start(self) -> None:
        """Active l'éditeur."""
        self.active = True
        self.points = []
        self.landing_zone_start = None
        self.landing_zone_end = None
        self.vessel_position = None
        self.mode = 'draw'

    def stop(self) -> None:
        """Désactive l'éditeur."""
        self.active = False

    def handle_click(self, x: int, y: int, button: int) -> None:
        """Gère un clic de souris."""
        if not self.active:
            return

        # Convertir en coordonnées monde
        world_x = x * self.echelle
        world_y = y * self.echelle

        if button == 1:  # Clic gauche
            if self.mode == 'draw':
                # Ajouter un point
                self.points.append((world_x, world_y))
            elif self.mode == 'landing_zone':
                if self.landing_zone_start is None:
                    self.landing_zone_start = len(self.points) - 1
                else:
                    self.landing_zone_end = len(self.points) - 1
                    self.mode = 'vessel'
            elif self.mode == 'vessel':
                self.vessel_position = (world_x, world_y)

        elif button == 3:  # Clic droit
            # Supprimer le dernier point
            if self.points:
                self.points.pop()

    def handle_key(self, key: int) -> str:
        """Gère une touche. Retourne un message."""
        if not self.active:
            return ""

        if key == pygame.K_RETURN:
            # Valider le terrain
            if len(self.points) >= 2:
                self.mode = 'landing_zone'
                return "Cliquez pour définir la zone d'atterrissage"

        elif key == pygame.K_s:
            # Sauvegarder
            return self.save_level()

        elif key == pygame.K_c:
            # Effacer
            self.points = []
            self.landing_zone_start = None
            self.landing_zone_end = None
            self.vessel_position = None
            self.mode = 'draw'
            return "Terrain effacé"

        elif key == pygame.K_ESCAPE:
            self.stop()
            return "Éditeur fermé"

        return ""

    def save_level(self, name: str = None) -> str:
        """Sauvegarde le niveau actuel."""
        if len(self.points) < 2:
            return "Pas assez de points"

        if name is None:
            name = f"custom_level_{len(self.saved_levels) + 1}"

        level = {
            'name': name,
            'surface_mars': self.points.copy(),
            'vaisseau': self.vessel_position or (self.screen_width * self.echelle // 2, 500),
            'landing_zone': (self.landing_zone_start, self.landing_zone_end)
        }

        self.saved_levels.append(level)

        # Sauvegarder sur disque
        try:
            with open(chemin_sauvegarde('custom_levels.pkl'), 'wb') as f:
                pickle.dump(self.saved_levels, f)
            return f"Niveau '{name}' sauvegardé!"
        except Exception as e:
            return f"Erreur: {e}"

    def load_levels(self) -> int:
        """Charge les niveaux sauvegardés."""
        try:
            if os.path.exists(chemin_sauvegarde('custom_levels.pkl')):
                with open(chemin_sauvegarde('custom_levels.pkl'), 'rb') as f:
                    self.saved_levels = pickle.load(f)
                return len(self.saved_levels)
        except:
            pass
        return 0

    def get_level(self, index: int = -1) -> Optional[Dict]:
        """Retourne un niveau sauvegardé."""
        if 0 <= index < len(self.saved_levels):
            return self.saved_levels[index]
        elif index == -1 and self.saved_levels:
            return self.saved_levels[-1]
        return None

    def get_current_as_scenario(self) -> Optional[Dict]:
        """Convertit le terrain actuel en scénario jouable."""
        if len(self.points) < 2:
            return None

        # Assurer que le terrain est valide
        surface = self.points.copy()

        # Position du vaisseau
        vessel_pos = self.vessel_position or (
            self.screen_width * self.echelle // 2,
            500
        )

        return {
            'surface_mars': surface,
            'vaisseau': {
                'x': vessel_pos[0],
                'y': vessel_pos[1],
                'h_speed': 0,
                'v_speed': 0,
                'fuel': 500,
                'angle': 0,
                'puissance': 0
            }
        }

    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        """Dessine l'éditeur."""
        if not self.active:
            return

        # Fond semi-transparent
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 30, 200))
        screen.blit(overlay, (0, 0))

        # Grille
        grid_color = (50, 50, 80)
        for x in range(0, self.screen_width, 50):
            pygame.draw.line(screen, grid_color, (x, 0), (x, self.screen_height))
        for y in range(0, self.screen_height, 50):
            pygame.draw.line(screen, grid_color, (0, y), (self.screen_width, y))

        # Points du terrain
        if len(self.points) >= 2:
            screen_points = [(p[0] // self.echelle, p[1] // self.echelle) for p in self.points]
            pygame.draw.lines(screen, (200, 100, 50), False, screen_points, 3)

        # Points individuels
        for i, p in enumerate(self.points):
            px, py = p[0] // self.echelle, p[1] // self.echelle
            color = (0, 255, 0) if self.landing_zone_start and self.landing_zone_end and \
                    self.landing_zone_start <= i <= self.landing_zone_end else (255, 255, 255)
            pygame.draw.circle(screen, color, (px, py), 5)

        # Position du vaisseau
        if self.vessel_position:
            vx, vy = self.vessel_position[0] // self.echelle, self.vessel_position[1] // self.echelle
            pygame.draw.polygon(screen, (0, 200, 255), [
                (vx, vy - 15), (vx - 10, vy + 10), (vx + 10, vy + 10)
            ])

        # Instructions
        instructions = [
            "ÉDITEUR DE NIVEAUX",
            f"Mode: {self.mode.upper()}",
            "",
            "Clic gauche: Ajouter point",
            "Clic droit: Supprimer point",
            "ENTRÉE: Valider terrain",
            "S: Sauvegarder",
            "C: Effacer",
            "ESC: Quitter"
        ]

        y = 10
        for line in instructions:
            text = font.render(line, True, (255, 255, 255))
            screen.blit(text, (self.screen_width - 200, y))
            y += 20


# =============================================================================
# 17. DASHBOARD STATISTIQUES
# =============================================================================

class StatsDashboard:
    """
    Dashboard de statistiques avancées.
    Affiche des graphiques et métriques détaillées.
    """

    def __init__(self, screen_width: int = 1000, screen_height: int = 600):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.enabled = False
        self.visible = False

        # Données
        self.episode_rewards: deque = deque(maxlen=1000)
        self.episode_durations: deque = deque(maxlen=1000)
        self.landing_positions: List[Tuple[float, float]] = []
        self.crash_positions: List[Tuple[float, float]] = []
        self.q_table_sizes: deque = deque(maxlen=1000)
        self.epsilon_history: deque = deque(maxlen=1000)

        # Statistiques globales
        self.total_episodes = 0
        self.total_landings = 0
        self.total_crashes = 0
        self.best_reward = float('-inf')
        self.best_landing_fuel = 0
        self.session_start_time = time.time()

    def toggle(self) -> None:
        """Affiche/cache le dashboard."""
        self.visible = not self.visible

    def record_episode(self, reward: float, duration: float, landed: bool,
                       position: Tuple[float, float], fuel: float = 0) -> None:
        """Enregistre les données d'un épisode."""
        self.episode_rewards.append(reward)
        self.episode_durations.append(duration)
        self.total_episodes += 1

        if landed:
            self.total_landings += 1
            self.landing_positions.append(position)
            if fuel > self.best_landing_fuel:
                self.best_landing_fuel = fuel
        else:
            self.total_crashes += 1
            self.crash_positions.append(position)

        if reward > self.best_reward:
            self.best_reward = reward

    def record_training_step(self, q_table_size: int, epsilon: float) -> None:
        """Enregistre les métriques d'entraînement."""
        self.q_table_sizes.append(q_table_size)
        self.epsilon_history.append(epsilon)

    def get_success_rate(self) -> float:
        """Retourne le taux de réussite."""
        if self.total_episodes == 0:
            return 0
        return self.total_landings / self.total_episodes

    def get_average_reward(self, n: int = 100) -> float:
        """Retourne la récompense moyenne des n derniers épisodes."""
        if not self.episode_rewards:
            return 0
        recent = list(self.episode_rewards)[-n:]
        return sum(recent) / len(recent)

    def draw_graph(self, screen: pygame.Surface, data: List[float],
                   rect: pygame.Rect, color: Tuple[int, int, int],
                   title: str, font: pygame.font.Font) -> None:
        """Dessine un graphique."""
        if not data:
            return

        # Fond
        pygame.draw.rect(screen, (30, 30, 40), rect)
        pygame.draw.rect(screen, (100, 100, 100), rect, 1)

        # Titre
        title_surf = font.render(title, True, (255, 255, 255))
        screen.blit(title_surf, (rect.x + 5, rect.y + 2))

        # Zone du graphique
        graph_rect = pygame.Rect(rect.x + 5, rect.y + 20,
                                rect.width - 10, rect.height - 25)

        # Normalisation
        min_val = min(data)
        max_val = max(data)
        if max_val == min_val:
            max_val = min_val + 1

        # Points
        points = []
        for i, val in enumerate(data):
            x = graph_rect.x + (i / max(1, len(data) - 1)) * graph_rect.width
            y = graph_rect.bottom - ((val - min_val) / (max_val - min_val)) * graph_rect.height
            points.append((x, y))

        # Ligne
        if len(points) >= 2:
            pygame.draw.lines(screen, color, False, points, 2)

        # Valeurs min/max
        min_text = font.render(f"{min_val:.1f}", True, (150, 150, 150))
        max_text = font.render(f"{max_val:.1f}", True, (150, 150, 150))
        screen.blit(min_text, (graph_rect.right - 40, graph_rect.bottom - 15))
        screen.blit(max_text, (graph_rect.right - 40, graph_rect.top))

    def draw_heatmap(self, screen: pygame.Surface, positions: List[Tuple[float, float]],
                     rect: pygame.Rect, color: Tuple[int, int, int],
                     title: str, font: pygame.font.Font, echelle: int = 10) -> None:
        """Dessine une heatmap des positions."""
        # Fond
        pygame.draw.rect(screen, (30, 30, 40), rect)
        pygame.draw.rect(screen, (100, 100, 100), rect, 1)

        # Titre
        title_surf = font.render(title, True, (255, 255, 255))
        screen.blit(title_surf, (rect.x + 5, rect.y + 2))

        # Points
        for x, y in positions[-200:]:  # Limiter aux 200 derniers
            px = rect.x + (x / (7000)) * rect.width
            py = rect.y + 20 + (y / (3000)) * (rect.height - 25)
            pygame.draw.circle(screen, (*color, 100), (int(px), int(py)), 3)

    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        """Dessine le dashboard complet."""
        if not self.visible:
            return

        # Fond semi-transparent
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 30, 230))
        screen.blit(overlay, (0, 0))

        # Titre
        title = font.render("DASHBOARD STATISTIQUES", True, (255, 215, 0))
        screen.blit(title, (self.screen_width // 2 - 100, 10))

        # Statistiques générales
        stats = [
            f"Épisodes: {self.total_episodes}",
            f"Atterrissages: {self.total_landings}",
            f"Crashes: {self.total_crashes}",
            f"Taux de réussite: {self.get_success_rate():.1%}",
            f"Meilleure récompense: {self.best_reward:.1f}",
            f"Meilleur fuel: {self.best_landing_fuel:.0f}",
            f"Récompense moy (100): {self.get_average_reward():.1f}",
        ]

        y = 40
        for stat in stats:
            text = font.render(stat, True, (200, 200, 200))
            screen.blit(text, (20, y))
            y += 22

        # Graphique des récompenses
        if self.episode_rewards:
            self.draw_graph(screen, list(self.episode_rewards),
                          pygame.Rect(20, 200, 300, 150),
                          (0, 255, 100), "Récompenses", font)

        # Graphique epsilon
        if self.epsilon_history:
            self.draw_graph(screen, list(self.epsilon_history),
                          pygame.Rect(340, 200, 300, 150),
                          (255, 255, 0), "Epsilon", font)

        # Graphique Q-table
        if self.q_table_sizes:
            self.draw_graph(screen, list(self.q_table_sizes),
                          pygame.Rect(660, 200, 300, 150),
                          (0, 200, 255), "Taille Q-Table", font)

        # Heatmap atterrissages
        if self.landing_positions:
            self.draw_heatmap(screen, self.landing_positions,
                            pygame.Rect(20, 370, 300, 200),
                            (0, 255, 0), "Atterrissages", font)

        # Heatmap crashes
        if self.crash_positions:
            self.draw_heatmap(screen, self.crash_positions,
                            pygame.Rect(340, 370, 300, 200),
                            (255, 0, 0), "Crashes", font)

        # Instructions
        inst = font.render("Appuyez sur D pour fermer", True, (150, 150, 150))
        screen.blit(inst, (self.screen_width // 2 - 100, self.screen_height - 25))

    def reset(self) -> None:
        """Réinitialise les statistiques."""
        self.episode_rewards.clear()
        self.episode_durations.clear()
        self.landing_positions.clear()
        self.crash_positions.clear()
        self.q_table_sizes.clear()
        self.epsilon_history.clear()
        self.total_episodes = 0
        self.total_landings = 0
        self.total_crashes = 0
        self.best_reward = float('-inf')
        self.best_landing_fuel = 0
        self.session_start_time = time.time()
