"""
Systèmes de jeu avancés pour Mars Lander.

Ce module contient les nouvelles fonctionnalités:
- Météorites dynamiques
- Système de dégâts progressifs
- Power-ups et bonus
- Stations de ravitaillement
- Zones d'atterrissage multiples
- Mode nuit et éclairage
- Tempêtes de poussière
- Brouillard de guerre
- Caméra dynamique
- Mode Time Attack
- Mode Survie
- Système de missions
"""

import pygame
import random
import math
import time
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field


# =============================================================================
# 1. MÉTÉORITES DYNAMIQUES
# =============================================================================

@dataclass
class Meteorite:
    """Représente une météorite tombant du ciel."""
    x: float
    y: float
    vx: float  # Vitesse horizontale
    vy: float  # Vitesse verticale
    radius: float
    color: Tuple[int, int, int] = (139, 69, 19)  # Marron
    active: bool = True
    rotation: float = 0
    rotation_speed: float = 0


class MeteoriteSystem:
    """
    Gère les météorites qui tombent du ciel.
    L'IA doit apprendre à les éviter.
    """

    def __init__(self, screen_width: int = 1000, screen_height: int = 600):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.meteorites: List[Meteorite] = []
        self.spawn_rate = 0.02  # Probabilité de spawn par frame
        self.enabled = True
        self.difficulty = 1.0  # Multiplicateur de difficulté

    def update(self) -> None:
        """Met à jour les météorites."""
        if not self.enabled:
            return

        # Spawn de nouvelles météorites
        if random.random() < self.spawn_rate * self.difficulty:
            self.spawn_meteorite()

        # Mise à jour des météorites existantes
        for m in self.meteorites:
            if m.active:
                m.x += m.vx
                m.y += m.vy
                m.vy += 0.1  # Gravité
                m.rotation += m.rotation_speed

                # Désactiver si hors écran
                if m.y > self.screen_height + 50 or m.x < -50 or m.x > self.screen_width + 50:
                    m.active = False

        # Nettoyer les météorites inactives
        self.meteorites = [m for m in self.meteorites if m.active]

    def spawn_meteorite(self) -> None:
        """Crée une nouvelle météorite."""
        x = random.randint(0, self.screen_width)
        y = -20
        vx = random.uniform(-2, 2)
        vy = random.uniform(2, 5) * self.difficulty
        radius = random.uniform(8, 20)
        rotation_speed = random.uniform(-0.1, 0.1)

        self.meteorites.append(Meteorite(
            x=x, y=y, vx=vx, vy=vy, radius=radius,
            rotation_speed=rotation_speed
        ))

    def check_collision(self, vessel_x: float, vessel_y: float, vessel_radius: float = 20) -> bool:
        """Vérifie si le vaisseau entre en collision avec une météorite."""
        for m in self.meteorites:
            if m.active:
                dist = math.sqrt((m.x - vessel_x)**2 + (m.y - vessel_y)**2)
                if dist < m.radius + vessel_radius:
                    m.active = False
                    return True
        return False

    def draw(self, screen: pygame.Surface) -> None:
        """Dessine les météorites."""
        for m in self.meteorites:
            if m.active:
                # Dessiner la météorite avec rotation
                points = []
                num_points = 7
                for i in range(num_points):
                    angle = m.rotation + (2 * math.pi * i / num_points)
                    r = m.radius * (0.7 + 0.3 * math.sin(i * 2.5))
                    px = m.x + r * math.cos(angle)
                    py = m.y + r * math.sin(angle)
                    points.append((px, py))

                pygame.draw.polygon(screen, m.color, points)
                pygame.draw.polygon(screen, (100, 50, 10), points, 2)

                # Traînée de feu
                trail_color = (255, 100, 0)
                for i in range(3):
                    trail_x = m.x - m.vx * (i + 1) * 3
                    trail_y = m.y - m.vy * (i + 1) * 2
                    trail_radius = m.radius * (0.5 - i * 0.1)
                    pygame.draw.circle(screen, trail_color, (int(trail_x), int(trail_y)), int(trail_radius))


# =============================================================================
# 2. SYSTÈME DE DÉGÂTS PROGRESSIFS
# =============================================================================

class DamageSystem:
    """
    Gère les dégâts progressifs du vaisseau.
    Le vaisseau peut subir des dommages partiels avant destruction totale.
    """

    def __init__(self):
        self.health = 100.0  # Points de vie (0-100)
        self.max_health = 100.0
        self.shield = 0.0  # Bouclier (0-100)
        self.max_shield = 100.0
        self.damage_multiplier = 1.0
        self.enabled = True

        # Seuils de dégâts visuels
        self.damage_levels = {
            75: "légèrement endommagé",
            50: "endommagé",
            25: "gravement endommagé",
            0: "détruit"
        }

    def take_damage(self, amount: float, source: str = "collision") -> Tuple[float, bool]:
        """
        Inflige des dégâts au vaisseau.

        Returns:
            (dégâts réels infligés, vaisseau détruit?)
        """
        if not self.enabled:
            return (0, False)

        actual_damage = amount * self.damage_multiplier

        # Le bouclier absorbe d'abord
        if self.shield > 0:
            shield_absorb = min(self.shield, actual_damage)
            self.shield -= shield_absorb
            actual_damage -= shield_absorb

        # Puis la santé
        self.health -= actual_damage
        self.health = max(0, self.health)

        return (actual_damage, self.health <= 0)

    def heal(self, amount: float) -> None:
        """Répare le vaisseau."""
        self.health = min(self.max_health, self.health + amount)

    def add_shield(self, amount: float) -> None:
        """Ajoute du bouclier."""
        self.shield = min(self.max_shield, self.shield + amount)

    def get_damage_level(self) -> str:
        """Retourne le niveau de dégâts actuel."""
        for threshold, level in sorted(self.damage_levels.items(), reverse=True):
            if self.health <= threshold:
                return level
        return "intact"

    def get_health_color(self) -> Tuple[int, int, int]:
        """Retourne une couleur selon la santé."""
        if self.health > 75:
            return (0, 255, 0)  # Vert
        elif self.health > 50:
            return (255, 255, 0)  # Jaune
        elif self.health > 25:
            return (255, 128, 0)  # Orange
        else:
            return (255, 0, 0)  # Rouge

    def reset(self) -> None:
        """Réinitialise les dégâts."""
        self.health = self.max_health
        self.shield = 0

    def draw_health_bar(self, screen: pygame.Surface, x: int, y: int, width: int = 100, height: int = 10) -> None:
        """Dessine la barre de vie."""
        # Fond
        pygame.draw.rect(screen, (50, 50, 50), (x, y, width, height))

        # Bouclier (bleu)
        if self.shield > 0:
            shield_width = int(width * self.shield / self.max_shield)
            pygame.draw.rect(screen, (0, 150, 255), (x, y - height - 2, shield_width, height))

        # Santé
        health_width = int(width * self.health / self.max_health)
        pygame.draw.rect(screen, self.get_health_color(), (x, y, health_width, height))

        # Bordure
        pygame.draw.rect(screen, (255, 255, 255), (x, y, width, height), 1)


# =============================================================================
# 3. POWER-UPS ET BONUS
# =============================================================================

@dataclass
class PowerUp:
    """Représente un power-up collectible."""
    x: float
    y: float
    type: str  # 'fuel', 'shield', 'repair', 'slowmo', 'magnet'
    radius: float = 15
    active: bool = True
    pulse: float = 0  # Animation de pulsation

    COLORS = {
        'fuel': (255, 200, 0),      # Jaune - Carburant
        'shield': (0, 150, 255),    # Bleu - Bouclier
        'repair': (0, 255, 0),      # Vert - Réparation
        'slowmo': (255, 0, 255),    # Magenta - Ralenti
        'magnet': (255, 100, 100),  # Rose - Aimant vers zone
        'points': (255, 255, 255),  # Blanc - Points bonus
    }

    ICONS = {
        'fuel': 'F',
        'shield': 'S',
        'repair': '+',
        'slowmo': 'T',
        'magnet': 'M',
        'points': '*',
    }


class PowerUpSystem:
    """Gère les power-ups dans le jeu."""

    def __init__(self, screen_width: int = 1000, screen_height: int = 600):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.powerups: List[PowerUp] = []
        self.spawn_rate = 0.005
        self.enabled = True
        self.active_effects: Dict[str, float] = {}  # type -> temps restant

    def update(self, dt: float = 1/60) -> None:
        """Met à jour les power-ups."""
        if not self.enabled:
            return

        # Spawn aléatoire
        if random.random() < self.spawn_rate:
            self.spawn_powerup()

        # Animation de pulsation
        for p in self.powerups:
            p.pulse = (p.pulse + 0.1) % (2 * math.pi)

        # Mise à jour des effets actifs
        expired = []
        for effect_type, time_left in self.active_effects.items():
            self.active_effects[effect_type] = time_left - dt
            if self.active_effects[effect_type] <= 0:
                expired.append(effect_type)
        for e in expired:
            del self.active_effects[e]

    def spawn_powerup(self, forced_type: str = None) -> None:
        """Crée un nouveau power-up."""
        x = random.randint(50, self.screen_width - 50)
        y = random.randint(100, self.screen_height - 150)

        if forced_type:
            ptype = forced_type
        else:
            ptype = random.choice(['fuel', 'shield', 'repair', 'slowmo', 'points'])

        self.powerups.append(PowerUp(x=x, y=y, type=ptype))

    def check_collection(self, vessel_x: float, vessel_y: float, vessel_radius: float = 25) -> Optional[str]:
        """Vérifie si le vaisseau collecte un power-up."""
        for p in self.powerups:
            if p.active:
                dist = math.sqrt((p.x - vessel_x)**2 + (p.y - vessel_y)**2)
                if dist < p.radius + vessel_radius:
                    p.active = False
                    return p.type

        # Nettoyer
        self.powerups = [p for p in self.powerups if p.active]
        return None

    def apply_effect(self, effect_type: str, vessel, damage_system: DamageSystem = None) -> str:
        """Applique l'effet d'un power-up."""
        message = ""

        if effect_type == 'fuel':
            vessel.fuel = min(vessel.fuel + 200, 1000)
            message = "+200 Fuel"
        elif effect_type == 'shield':
            if damage_system:
                damage_system.add_shield(50)
            message = "+50 Shield"
        elif effect_type == 'repair':
            if damage_system:
                damage_system.heal(30)
            message = "+30 HP"
        elif effect_type == 'slowmo':
            self.active_effects['slowmo'] = 5.0  # 5 secondes
            message = "Slow Motion 5s"
        elif effect_type == 'points':
            message = "+500 Points"
        elif effect_type == 'magnet':
            self.active_effects['magnet'] = 8.0
            message = "Magnet 8s"

        return message

    def is_slowmo_active(self) -> bool:
        """Vérifie si le ralenti est actif."""
        return 'slowmo' in self.active_effects

    def draw(self, screen: pygame.Surface, font: pygame.font.Font = None) -> None:
        """Dessine les power-ups."""
        for p in self.powerups:
            if p.active:
                # Pulsation
                pulse_scale = 1 + 0.2 * math.sin(p.pulse)
                radius = int(p.radius * pulse_scale)

                # Cercle principal
                color = PowerUp.COLORS.get(p.type, (255, 255, 255))
                pygame.draw.circle(screen, color, (int(p.x), int(p.y)), radius)
                pygame.draw.circle(screen, (255, 255, 255), (int(p.x), int(p.y)), radius, 2)

                # Icône
                if font:
                    icon = PowerUp.ICONS.get(p.type, '?')
                    text = font.render(icon, True, (0, 0, 0))
                    text_rect = text.get_rect(center=(int(p.x), int(p.y)))
                    screen.blit(text, text_rect)

                # Aura
                aura_radius = int(radius * 1.5)
                aura_surface = pygame.Surface((aura_radius * 2, aura_radius * 2), pygame.SRCALPHA)
                aura_alpha = int(50 + 30 * math.sin(p.pulse))
                pygame.draw.circle(aura_surface, (*color, aura_alpha), (aura_radius, aura_radius), aura_radius)
                screen.blit(aura_surface, (int(p.x) - aura_radius, int(p.y) - aura_radius))


# =============================================================================
# 4. STATIONS DE RAVITAILLEMENT
# =============================================================================

@dataclass
class FuelStation:
    """Station de ravitaillement en vol."""
    x: float
    y: float
    fuel_amount: float = 300
    max_fuel: float = 300
    recharge_rate: float = 0.5  # Fuel régénéré par frame
    radius: float = 30
    active: bool = True
    docking_range: float = 50


class FuelStationSystem:
    """Gère les stations de ravitaillement."""

    def __init__(self):
        self.stations: List[FuelStation] = []
        self.enabled = True
        self.refuel_rate = 2.0  # Fuel par frame quand docké

    def add_station(self, x: float, y: float) -> None:
        """Ajoute une station."""
        self.stations.append(FuelStation(x=x, y=y))

    def update(self) -> None:
        """Met à jour les stations (régénération du fuel)."""
        for station in self.stations:
            if station.fuel_amount < station.max_fuel:
                station.fuel_amount = min(station.max_fuel,
                                         station.fuel_amount + station.recharge_rate)

    def check_docking(self, vessel_x: float, vessel_y: float, vessel) -> Tuple[bool, float]:
        """
        Vérifie si le vaisseau est docké et ravitaille.
        Returns: (est_docké, fuel_transféré)
        """
        if not self.enabled:
            return (False, 0)

        for station in self.stations:
            dist = math.sqrt((station.x - vessel_x)**2 + (station.y - vessel_y)**2)

            if dist < station.docking_range and station.fuel_amount > 0:
                # Calculer le fuel à transférer
                fuel_needed = 1000 - vessel.fuel  # Capacité max supposée 1000
                fuel_available = min(station.fuel_amount, self.refuel_rate)
                fuel_transfer = min(fuel_needed, fuel_available)

                if fuel_transfer > 0:
                    station.fuel_amount -= fuel_transfer
                    vessel.fuel += fuel_transfer
                    return (True, fuel_transfer)

        return (False, 0)

    def draw(self, screen: pygame.Surface, font: pygame.font.Font = None) -> None:
        """Dessine les stations."""
        for station in self.stations:
            # Plateforme
            platform_color = (100, 100, 150) if station.fuel_amount > 0 else (80, 80, 80)
            pygame.draw.rect(screen, platform_color,
                           (station.x - 25, station.y - 5, 50, 10))

            # Pylône
            pygame.draw.rect(screen, (80, 80, 80),
                           (station.x - 3, station.y - 30, 6, 25))

            # Indicateur de fuel
            fuel_ratio = station.fuel_amount / station.max_fuel
            fuel_color = (0, 255, 0) if fuel_ratio > 0.5 else (255, 255, 0) if fuel_ratio > 0.2 else (255, 0, 0)
            pygame.draw.rect(screen, (50, 50, 50), (station.x - 15, station.y - 45, 30, 8))
            pygame.draw.rect(screen, fuel_color, (station.x - 15, station.y - 45, int(30 * fuel_ratio), 8))

            # Zone de docking (cercle)
            pygame.draw.circle(screen, (0, 255, 0, 50), (int(station.x), int(station.y)),
                             int(station.docking_range), 1)

            # Texte
            if font and station.fuel_amount > 0:
                text = font.render(f"{int(station.fuel_amount)}", True, (255, 255, 255))
                screen.blit(text, (station.x - 15, station.y - 60))


# =============================================================================
# 5. ZONES D'ATTERRISSAGE MULTIPLES
# =============================================================================

@dataclass
class LandingZone:
    """Zone d'atterrissage avec sa propre récompense."""
    x1: float
    x2: float
    y: float
    points: int
    difficulty: str  # 'easy', 'medium', 'hard', 'expert'
    name: str = ""

    COLORS = {
        'easy': (0, 255, 0),      # Vert - Facile
        'medium': (255, 255, 0),  # Jaune - Moyen
        'hard': (255, 128, 0),    # Orange - Difficile
        'expert': (255, 0, 0),    # Rouge - Expert
    }


class MultiZoneSystem:
    """Gère plusieurs zones d'atterrissage."""

    def __init__(self):
        self.zones: List[LandingZone] = []
        self.enabled = True

    def add_zone(self, x1: float, x2: float, y: float, points: int,
                 difficulty: str, name: str = "") -> None:
        """Ajoute une zone d'atterrissage."""
        self.zones.append(LandingZone(x1=x1, x2=x2, y=y, points=points,
                                      difficulty=difficulty, name=name))

    def check_landing(self, vessel_x: float, vessel_y: float,
                      tolerance: float = 30) -> Optional[LandingZone]:
        """Vérifie dans quelle zone le vaisseau a atterri."""
        for zone in self.zones:
            if zone.x1 <= vessel_x <= zone.x2 and abs(vessel_y - zone.y) < tolerance:
                return zone
        return None

    def get_nearest_zone(self, vessel_x: float, vessel_y: float) -> Tuple[Optional[LandingZone], float]:
        """Retourne la zone la plus proche et la distance."""
        nearest = None
        min_dist = float('inf')

        for zone in self.zones:
            zone_center_x = (zone.x1 + zone.x2) / 2
            dist = math.sqrt((zone_center_x - vessel_x)**2 + (zone.y - vessel_y)**2)
            if dist < min_dist:
                min_dist = dist
                nearest = zone

        return (nearest, min_dist)

    def draw(self, screen: pygame.Surface, font: pygame.font.Font = None, echelle: int = 1) -> None:
        """Dessine les indicateurs de zones."""
        for zone in self.zones:
            color = LandingZone.COLORS.get(zone.difficulty, (255, 255, 255))

            # Ligne de la zone
            x1 = int(zone.x1 / echelle)
            x2 = int(zone.x2 / echelle)
            y = int(zone.y / echelle)

            pygame.draw.line(screen, color, (x1, y - 5), (x2, y - 5), 3)

            # Points
            if font:
                points_text = font.render(f"{zone.points}pts", True, color)
                center_x = (x1 + x2) // 2
                screen.blit(points_text, (center_x - 20, y - 25))


# =============================================================================
# 6. MODE NUIT ET ÉCLAIRAGE
# =============================================================================

class NightModeSystem:
    """Gère le mode nuit avec éclairage du vaisseau."""

    def __init__(self, screen_width: int = 1000, screen_height: int = 600):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.enabled = False
        self.darkness_level = 200  # 0-255 (255 = noir total)
        self.spotlight_radius = 150
        self.spotlight_intensity = 1.0
        self.ambient_light = 30  # Lumière ambiante minimale

        # Surface pour l'obscurité
        self.darkness_surface = None

    def toggle(self) -> None:
        """Active/désactive le mode nuit."""
        self.enabled = not self.enabled

    def create_spotlight_mask(self, vessel_x: float, vessel_y: float,
                              vessel_angle: float = 0) -> pygame.Surface:
        """Crée le masque de lumière."""
        if self.darkness_surface is None:
            self.darkness_surface = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)

        # Remplir avec l'obscurité
        self.darkness_surface.fill((0, 0, 0, self.darkness_level))

        # Créer le spot lumineux (cercle avec gradient)
        for r in range(self.spotlight_radius, 0, -5):
            alpha = int(self.darkness_level * (1 - (r / self.spotlight_radius) ** 0.5))
            pygame.draw.circle(self.darkness_surface, (0, 0, 0, alpha),
                             (int(vessel_x), int(vessel_y)), r)

        # Zone centrale complètement éclairée
        pygame.draw.circle(self.darkness_surface, (0, 0, 0, 0),
                         (int(vessel_x), int(vessel_y)), 30)

        return self.darkness_surface

    def draw(self, screen: pygame.Surface, vessel_x: float, vessel_y: float,
             vessel_angle: float = 0) -> None:
        """Applique l'effet de nuit."""
        if not self.enabled:
            return

        mask = self.create_spotlight_mask(vessel_x, vessel_y, vessel_angle)
        screen.blit(mask, (0, 0))


# =============================================================================
# 7. TEMPÊTES DE POUSSIÈRE
# =============================================================================

@dataclass
class DustParticle:
    """Particule de poussière."""
    x: float
    y: float
    vx: float
    vy: float
    size: float
    alpha: int
    lifetime: float


class DustStormSystem:
    """Gère les tempêtes de poussière."""

    def __init__(self, screen_width: int = 1000, screen_height: int = 600):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.particles: List[DustParticle] = []
        self.enabled = False
        self.storm_active = False
        self.storm_intensity = 0.0  # 0-1
        self.storm_duration = 0.0
        self.storm_wind = (0.0, 0.0)
        self.visibility = 1.0  # 0-1 (1 = pleine visibilité)
        self.max_particles = 300

    def start_storm(self, duration: float = 10.0, intensity: float = 0.7) -> None:
        """Démarre une tempête."""
        self.storm_active = True
        self.storm_duration = duration
        self.storm_intensity = intensity
        self.storm_wind = (random.uniform(-3, 3), random.uniform(0.5, 2))
        self.visibility = 1.0 - intensity * 0.6

    def update(self, dt: float = 1/60) -> None:
        """Met à jour la tempête."""
        if not self.enabled:
            return

        if self.storm_active:
            self.storm_duration -= dt
            if self.storm_duration <= 0:
                self.storm_active = False
                self.storm_intensity = 0
                self.visibility = 1.0

            # Spawn de particules
            particles_to_spawn = int(self.storm_intensity * 10)
            for _ in range(particles_to_spawn):
                if len(self.particles) < self.max_particles:
                    self.particles.append(DustParticle(
                        x=random.randint(-50, self.screen_width + 50),
                        y=random.randint(-50, self.screen_height),
                        vx=self.storm_wind[0] + random.uniform(-1, 1),
                        vy=self.storm_wind[1] + random.uniform(-0.5, 0.5),
                        size=random.uniform(2, 6),
                        alpha=random.randint(100, 200),
                        lifetime=random.uniform(2, 5)
                    ))

        # Mise à jour des particules
        for p in self.particles:
            p.x += p.vx
            p.y += p.vy
            p.lifetime -= dt
            p.alpha = max(0, p.alpha - 1)

        # Nettoyer
        self.particles = [p for p in self.particles
                         if p.lifetime > 0 and 0 < p.y < self.screen_height + 50]

    def get_wind_effect(self) -> Tuple[float, float]:
        """Retourne l'effet du vent de la tempête sur le vaisseau."""
        if self.storm_active:
            return (self.storm_wind[0] * 0.1, self.storm_wind[1] * 0.05)
        return (0, 0)

    def draw(self, screen: pygame.Surface) -> None:
        """Dessine la tempête."""
        if not self.enabled or not self.storm_active:
            return

        # Particules
        for p in self.particles:
            color = (180, 140, 100, p.alpha)
            surf = pygame.Surface((int(p.size * 2), int(p.size * 2)), pygame.SRCALPHA)
            pygame.draw.circle(surf, color, (int(p.size), int(p.size)), int(p.size))
            screen.blit(surf, (int(p.x - p.size), int(p.y - p.size)))

        # Overlay de visibilité réduite
        if self.visibility < 1.0:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay_alpha = int((1 - self.visibility) * 150)
            overlay.fill((180, 140, 100, overlay_alpha))
            screen.blit(overlay, (0, 0))


# =============================================================================
# 8. BROUILLARD DE GUERRE
# =============================================================================

class FogOfWarSystem:
    """Le terrain se révèle progressivement."""

    def __init__(self, screen_width: int = 1000, screen_height: int = 600):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.enabled = False
        self.revealed_areas: List[Tuple[float, float, float]] = []  # (x, y, radius)
        self.fog_surface = None
        self.fog_alpha = 220
        self.reveal_radius = 100

    def reveal(self, x: float, y: float, radius: float = None) -> None:
        """Révèle une zone."""
        if radius is None:
            radius = self.reveal_radius
        self.revealed_areas.append((x, y, radius))

    def update(self, vessel_x: float, vessel_y: float) -> None:
        """Met à jour le brouillard selon la position du vaisseau."""
        if not self.enabled:
            return
        self.reveal(vessel_x, vessel_y)

    def create_fog_mask(self) -> pygame.Surface:
        """Crée le masque de brouillard."""
        if self.fog_surface is None:
            self.fog_surface = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)

        # Remplir avec le brouillard
        self.fog_surface.fill((50, 50, 50, self.fog_alpha))

        # Révéler les zones explorées
        for x, y, radius in self.revealed_areas:
            # Gradient de révélation
            for r in range(int(radius), 0, -5):
                alpha = int(self.fog_alpha * (r / radius) ** 2)
                pygame.draw.circle(self.fog_surface, (50, 50, 50, alpha),
                                 (int(x), int(y)), r)
            pygame.draw.circle(self.fog_surface, (0, 0, 0, 0),
                             (int(x), int(y)), int(radius * 0.3))

        return self.fog_surface

    def draw(self, screen: pygame.Surface) -> None:
        """Dessine le brouillard."""
        if not self.enabled:
            return

        mask = self.create_fog_mask()
        screen.blit(mask, (0, 0))

    def reset(self) -> None:
        """Réinitialise le brouillard."""
        self.revealed_areas = []


# =============================================================================
# 9. CAMÉRA DYNAMIQUE
# =============================================================================

class DynamicCamera:
    """Caméra avec zoom et suivi du vaisseau."""

    def __init__(self, screen_width: int = 1000, screen_height: int = 600):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.enabled = False

        # Position et zoom
        self.x = 0
        self.y = 0
        self.zoom = 1.0
        self.target_zoom = 1.0
        self.min_zoom = 0.5
        self.max_zoom = 2.0

        # Paramètres de suivi
        self.follow_speed = 0.1
        self.zoom_speed = 0.05
        self.auto_zoom = True  # Zoom automatique selon altitude

    def update(self, vessel_x: float, vessel_y: float, vessel_vy: float = 0) -> None:
        """Met à jour la caméra."""
        if not self.enabled:
            return

        # Calcul de la position cible (centrer sur le vaisseau)
        target_x = vessel_x - self.screen_width / (2 * self.zoom)
        target_y = vessel_y - self.screen_height / (2 * self.zoom)

        # Interpolation douce
        self.x += (target_x - self.x) * self.follow_speed
        self.y += (target_y - self.y) * self.follow_speed

        # Auto-zoom selon l'altitude
        if self.auto_zoom:
            # Plus le vaisseau est haut, plus on dézoome
            altitude_ratio = vessel_y / self.screen_height
            self.target_zoom = 1.0 + (1 - altitude_ratio) * 0.5
            self.target_zoom = max(self.min_zoom, min(self.max_zoom, self.target_zoom))

        # Interpolation du zoom
        self.zoom += (self.target_zoom - self.zoom) * self.zoom_speed

    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convertit des coordonnées monde en coordonnées écran."""
        screen_x = (x - self.x) * self.zoom
        screen_y = (y - self.y) * self.zoom
        return (int(screen_x), int(screen_y))

    def screen_to_world(self, x: float, y: float) -> Tuple[float, float]:
        """Convertit des coordonnées écran en coordonnées monde."""
        world_x = x / self.zoom + self.x
        world_y = y / self.zoom + self.y
        return (world_x, world_y)

    def set_zoom(self, zoom: float) -> None:
        """Définit le zoom cible."""
        self.target_zoom = max(self.min_zoom, min(self.max_zoom, zoom))

    def reset(self) -> None:
        """Réinitialise la caméra."""
        self.x = 0
        self.y = 0
        self.zoom = 1.0
        self.target_zoom = 1.0


# =============================================================================
# 10. MODE TIME ATTACK
# =============================================================================

class TimeAttackMode:
    """Mode Time Attack avec chronomètre."""

    def __init__(self):
        self.enabled = False
        self.start_time = 0
        self.end_time = 0
        self.best_time = float('inf')
        self.running = False
        self.times_history: List[float] = []

    def start(self) -> None:
        """Démarre le chronomètre."""
        self.start_time = time.time()
        self.running = True
        self.end_time = 0

    def stop(self, success: bool = True) -> float:
        """Arrête le chronomètre."""
        if self.running:
            self.end_time = time.time()
            self.running = False
            elapsed = self.end_time - self.start_time

            if success:
                self.times_history.append(elapsed)
                if elapsed < self.best_time:
                    self.best_time = elapsed

            return elapsed
        return 0

    def get_elapsed(self) -> float:
        """Retourne le temps écoulé."""
        if self.running:
            return time.time() - self.start_time
        elif self.end_time > 0:
            return self.end_time - self.start_time
        return 0

    def get_best_time(self) -> float:
        """Retourne le meilleur temps."""
        return self.best_time if self.best_time != float('inf') else 0

    def format_time(self, seconds: float) -> str:
        """Formate le temps en mm:ss.ms."""
        if seconds == float('inf') or seconds == 0:
            return "--:--.--"
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02d}:{secs:05.2f}"

    def draw(self, screen: pygame.Surface, font: pygame.font.Font, x: int = 10, y: int = 10) -> None:
        """Affiche le chronomètre."""
        if not self.enabled:
            return

        # Temps actuel
        elapsed = self.get_elapsed()
        time_text = self.format_time(elapsed)
        color = (255, 255, 0) if self.running else (0, 255, 0)

        text = font.render(f"Time: {time_text}", True, color)
        screen.blit(text, (x, y))

        # Meilleur temps
        best = self.format_time(self.best_time)
        best_text = font.render(f"Best: {best}", True, (255, 215, 0))
        screen.blit(best_text, (x, y + 25))


# =============================================================================
# 11. MODE SURVIE
# =============================================================================

class SurvivalMode:
    """Mode survie avec vies et difficulté croissante."""

    def __init__(self):
        self.enabled = False
        self.lives = 3
        self.max_lives = 3
        self.score = 0
        self.level = 1
        self.landings_this_level = 0
        self.landings_per_level = 3
        self.difficulty_multiplier = 1.0
        self.game_over = False

    def start(self) -> None:
        """Démarre le mode survie."""
        self.lives = self.max_lives
        self.score = 0
        self.level = 1
        self.landings_this_level = 0
        self.difficulty_multiplier = 1.0
        self.game_over = False

    def on_landing(self, fuel_remaining: float, time_taken: float = 0) -> int:
        """Appelé lors d'un atterrissage réussi."""
        # Calcul du score
        base_score = 1000
        fuel_bonus = fuel_remaining * 2
        level_bonus = self.level * 500
        points = int((base_score + fuel_bonus + level_bonus) * self.difficulty_multiplier)

        self.score += points
        self.landings_this_level += 1

        # Passage au niveau suivant
        if self.landings_this_level >= self.landings_per_level:
            self.level_up()

        return points

    def on_crash(self) -> bool:
        """Appelé lors d'un crash. Retourne True si game over."""
        self.lives -= 1
        if self.lives <= 0:
            self.game_over = True
        return self.game_over

    def level_up(self) -> None:
        """Passe au niveau suivant."""
        self.level += 1
        self.landings_this_level = 0
        self.difficulty_multiplier += 0.2

    def get_difficulty_settings(self) -> Dict[str, float]:
        """Retourne les paramètres de difficulté pour le niveau actuel."""
        return {
            'wind_strength': 0.3 * self.difficulty_multiplier,
            'meteorite_rate': 0.02 * self.difficulty_multiplier,
            'storm_chance': 0.1 * (self.level - 1),
            'fuel_start': max(300, 500 - self.level * 30),
        }

    def draw(self, screen: pygame.Surface, font: pygame.font.Font, x: int = 10, y: int = 10) -> None:
        """Affiche les infos du mode survie."""
        if not self.enabled:
            return

        # Vies (coeurs)
        lives_text = "Lives: " + "♥" * self.lives + "♡" * (self.max_lives - self.lives)
        text = font.render(lives_text, True, (255, 0, 0))
        screen.blit(text, (x, y))

        # Score
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        screen.blit(score_text, (x, y + 25))

        # Niveau
        level_text = font.render(f"Level: {self.level} ({self.landings_this_level}/{self.landings_per_level})",
                                True, (255, 215, 0))
        screen.blit(level_text, (x, y + 50))


# =============================================================================
# 12. SYSTÈME DE MISSIONS
# =============================================================================

@dataclass
class Mission:
    """Une mission avec objectifs."""
    id: str
    name: str
    description: str
    objectives: List[Dict[str, Any]]
    reward_points: int
    completed: bool = False
    progress: Dict[str, float] = field(default_factory=dict)


class MissionSystem:
    """Gère les missions et objectifs."""

    def __init__(self):
        self.missions: List[Mission] = []
        self.active_mission: Optional[Mission] = None
        self.completed_missions: List[str] = []
        self.enabled = True
        self.total_points = 0

        # Créer les missions par défaut
        self._create_default_missions()

    def _create_default_missions(self) -> None:
        """Crée les missions de base."""
        self.missions = [
            Mission(
                id="first_landing",
                name="Premier Pas",
                description="Atterrir avec succès",
                objectives=[{"type": "land", "count": 1}],
                reward_points=500
            ),
            Mission(
                id="fuel_saver",
                name="Économe",
                description="Atterrir avec plus de 50% de fuel",
                objectives=[{"type": "land_with_fuel", "min_fuel_percent": 50}],
                reward_points=1000
            ),
            Mission(
                id="speed_demon",
                name="Vitesse Lumière",
                description="Atterrir en moins de 30 secondes",
                objectives=[{"type": "land_in_time", "max_time": 30}],
                reward_points=1500
            ),
            Mission(
                id="precision",
                name="Précision",
                description="Atterrir sur une zone difficile",
                objectives=[{"type": "land_zone", "difficulty": "hard"}],
                reward_points=2000
            ),
            Mission(
                id="survivor",
                name="Survivant",
                description="Réussir 5 atterrissages d'affilée",
                objectives=[{"type": "consecutive_landings", "count": 5}],
                reward_points=3000
            ),
            Mission(
                id="storm_rider",
                name="Chasseur de Tempêtes",
                description="Atterrir pendant une tempête",
                objectives=[{"type": "land_in_storm"}],
                reward_points=2500
            ),
            Mission(
                id="collector",
                name="Collectionneur",
                description="Collecter 10 power-ups",
                objectives=[{"type": "collect_powerups", "count": 10}],
                reward_points=1500
            ),
            Mission(
                id="night_owl",
                name="Oiseau de Nuit",
                description="Atterrir 3 fois en mode nuit",
                objectives=[{"type": "land_night", "count": 3}],
                reward_points=2000
            ),
        ]

    def set_active_mission(self, mission_id: str) -> bool:
        """Active une mission."""
        for mission in self.missions:
            if mission.id == mission_id and not mission.completed:
                self.active_mission = mission
                mission.progress = {}
                return True
        return False

    def check_objective(self, event_type: str, event_data: Dict[str, Any] = None) -> bool:
        """Vérifie si un objectif est atteint."""
        if not self.active_mission:
            return False

        if event_data is None:
            event_data = {}

        for obj in self.active_mission.objectives:
            if obj["type"] == event_type:
                if event_type == "land":
                    count = self.active_mission.progress.get("land_count", 0) + 1
                    self.active_mission.progress["land_count"] = count
                    if count >= obj.get("count", 1):
                        return self._complete_mission()

                elif event_type == "land_with_fuel":
                    fuel_percent = event_data.get("fuel_percent", 0)
                    if fuel_percent >= obj.get("min_fuel_percent", 0):
                        return self._complete_mission()

                elif event_type == "land_in_time":
                    time_taken = event_data.get("time", float('inf'))
                    if time_taken <= obj.get("max_time", 0):
                        return self._complete_mission()

                elif event_type == "collect_powerups":
                    count = self.active_mission.progress.get("powerup_count", 0) + 1
                    self.active_mission.progress["powerup_count"] = count
                    if count >= obj.get("count", 1):
                        return self._complete_mission()

                elif event_type == "consecutive_landings":
                    count = self.active_mission.progress.get("consecutive", 0) + 1
                    self.active_mission.progress["consecutive"] = count
                    if count >= obj.get("count", 1):
                        return self._complete_mission()

        return False

    def _complete_mission(self) -> bool:
        """Marque la mission comme complétée."""
        if self.active_mission:
            self.active_mission.completed = True
            self.completed_missions.append(self.active_mission.id)
            self.total_points += self.active_mission.reward_points
            self.active_mission = None
            return True
        return False

    def on_crash(self) -> None:
        """Appelé lors d'un crash (reset des objectifs consécutifs)."""
        if self.active_mission:
            self.active_mission.progress["consecutive"] = 0

    def get_available_missions(self) -> List[Mission]:
        """Retourne les missions disponibles."""
        return [m for m in self.missions if not m.completed]

    def draw(self, screen: pygame.Surface, font: pygame.font.Font, x: int = 10, y: int = 10) -> None:
        """Affiche la mission active."""
        if not self.enabled or not self.active_mission:
            return

        # Fond semi-transparent
        bg = pygame.Surface((250, 60), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 150))
        screen.blit(bg, (x, y))

        # Titre de la mission
        title = font.render(f"Mission: {self.active_mission.name}", True, (255, 215, 0))
        screen.blit(title, (x + 5, y + 5))

        # Description
        desc = font.render(self.active_mission.description, True, (200, 200, 200))
        screen.blit(desc, (x + 5, y + 25))

        # Récompense
        reward = font.render(f"+{self.active_mission.reward_points} pts", True, (0, 255, 0))
        screen.blit(reward, (x + 5, y + 42))
