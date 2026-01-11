# -*- coding: utf-8 -*-
"""
Module graphismes avances PRO
Effets visuels spectaculaires avec:
- Terrain procedural (opensimplex)
- Systeme de particules avance
- Effets de lumiere et glow
- Nebuleuses et etoiles avec parallax
- Acceleration Numba
"""

import pygame
import numpy as np
import math
import random
from typing import List, Tuple, Optional
from opensimplex import OpenSimplex

# Tentative d'import numba pour acceleration
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
    print("[ProGraphics] Numba JIT disponible - acceleration activee")
except ImportError:
    NUMBA_AVAILABLE = False
    print("[ProGraphics] Numba non disponible - mode standard")
    # Decorateur factice
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


# ============================================================================
# TERRAIN PROCEDURAL
# ============================================================================

class ProceduralTerrain:
    """
    Generateur de terrain procedural avec OpenSimplex noise
    """

    def __init__(self, width: int = 7000, seed: int = None):
        self.width = width
        self.seed = seed if seed else random.randint(0, 1000000)
        self.noise = OpenSimplex(seed=self.seed)

        # Parametres du terrain
        self.base_height = 500
        self.amplitude = 800
        self.frequency = 0.001
        self.octaves = 4
        self.persistence = 0.5

        # Cache du terrain
        self.terrain_points = []
        self.landing_zones = []

    def generate(self, num_landing_zones: int = 2) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, int]]]:
        """
        Genere un terrain avec zones d'atterrissage
        Retourne (points_terrain, zones_atterrissage)
        """
        self.terrain_points = []
        self.landing_zones = []

        # Generation du bruit multi-octaves
        step = 50
        heights = []

        for x in range(0, self.width + step, step):
            height = self._fractal_noise(x)
            heights.append((x, height))

        # Creation des zones d'atterrissage plates
        zone_positions = []
        zone_width = 300

        for i in range(num_landing_zones):
            # Position aleatoire mais espacee
            min_x = (self.width // (num_landing_zones + 1)) * (i + 1) - zone_width
            max_x = min_x + zone_width * 2
            zone_x = random.randint(int(min_x), int(max_x))
            zone_positions.append(zone_x)

        # Aplatir les zones d'atterrissage
        for zone_x in zone_positions:
            # Trouver la hauteur moyenne de la zone
            zone_heights = [h for x, h in heights if zone_x - zone_width//2 < x < zone_x + zone_width//2]
            if zone_heights:
                flat_height = sum(zone_heights) / len(zone_heights)
                # Remplacer les hauteurs dans la zone
                heights = [(x, flat_height if zone_x - zone_width//2 < x < zone_x + zone_width//2 else h)
                          for x, h in heights]
                self.landing_zones.append((zone_x - zone_width//2, zone_x + zone_width//2, int(flat_height)))

        self.terrain_points = [(int(x), int(h)) for x, h in heights]
        return self.terrain_points, self.landing_zones

    def _fractal_noise(self, x: float) -> float:
        """Bruit fractal multi-octaves"""
        value = 0
        amplitude = self.amplitude
        frequency = self.frequency

        for _ in range(self.octaves):
            value += self.noise.noise2(x * frequency, 0) * amplitude
            amplitude *= self.persistence
            frequency *= 2

        return self.base_height + value

    def regenerate(self, new_seed: int = None):
        """Regenere avec une nouvelle seed"""
        self.seed = new_seed if new_seed else random.randint(0, 1000000)
        self.noise = OpenSimplex(seed=self.seed)
        return self.generate()


# ============================================================================
# SYSTEME DE PARTICULES AVANCE
# ============================================================================

class AdvancedParticle:
    """Particule avec physique avancee"""

    __slots__ = ['x', 'y', 'vx', 'vy', 'life', 'max_life', 'size', 'color',
                 'decay', 'gravity', 'type', 'trail', 'glow']

    def __init__(self, x, y, vx, vy, life, size, color, particle_type='default', glow=False):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.life = life
        self.max_life = life
        self.size = size
        self.color = color
        self.decay = 0.98
        self.gravity = 0.1
        self.type = particle_type
        self.trail = []
        self.glow = glow


class AdvancedParticleSystem:
    """
    Systeme de particules haute performance
    """

    def __init__(self, max_particles: int = 5000):
        self.particles: List[AdvancedParticle] = []
        self.max_particles = max_particles

        # Presets de particules
        self.presets = {
            'flame': {
                'colors': [(255, 200, 50), (255, 150, 30), (255, 100, 20), (200, 50, 10)],
                'size_range': (3, 8),
                'life_range': (20, 40),
                'speed_range': (2, 5),
                'glow': True
            },
            'smoke': {
                'colors': [(100, 100, 100), (80, 80, 80), (60, 60, 60)],
                'size_range': (5, 15),
                'life_range': (40, 80),
                'speed_range': (0.5, 2),
                'glow': False
            },
            'explosion': {
                'colors': [(255, 255, 200), (255, 200, 100), (255, 150, 50), (255, 100, 0)],
                'size_range': (4, 12),
                'life_range': (30, 60),
                'speed_range': (5, 15),
                'glow': True
            },
            'sparks': {
                'colors': [(255, 255, 100), (255, 200, 50)],
                'size_range': (1, 3),
                'life_range': (10, 30),
                'speed_range': (8, 15),
                'glow': True
            },
            'dust': {
                'colors': [(180, 140, 100), (160, 120, 80), (140, 100, 60)],
                'size_range': (2, 6),
                'life_range': (60, 120),
                'speed_range': (1, 3),
                'glow': False
            },
            'magic': {
                'colors': [(100, 200, 255), (150, 220, 255), (200, 240, 255)],
                'size_range': (2, 5),
                'life_range': (30, 50),
                'speed_range': (1, 4),
                'glow': True
            },
            'celebration': {
                'colors': [(255, 215, 0), (255, 255, 255), (0, 255, 100), (255, 100, 200)],
                'size_range': (3, 8),
                'life_range': (50, 100),
                'speed_range': (3, 8),
                'glow': True
            }
        }

    def emit(self, x: float, y: float, preset: str, count: int = 20,
             direction: float = None, spread: float = math.pi):
        """
        Emet des particules
        direction: angle en radians (None = omnidirectionnel)
        spread: angle de dispersion
        """
        if len(self.particles) >= self.max_particles:
            # Supprimer les plus vieilles
            self.particles = self.particles[count:]

        settings = self.presets.get(preset, self.presets['flame'])

        for _ in range(count):
            if direction is None:
                angle = random.uniform(0, 2 * math.pi)
            else:
                angle = direction + random.uniform(-spread/2, spread/2)

            speed = random.uniform(*settings['speed_range'])
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed

            life = random.randint(*settings['life_range'])
            size = random.uniform(*settings['size_range'])
            color = random.choice(settings['colors'])

            particle = AdvancedParticle(
                x, y, vx, vy, life, size, color,
                preset, settings['glow']
            )

            self.particles.append(particle)

    def emit_trail(self, x: float, y: float, vx: float, vy: float, preset: str = 'flame'):
        """Emet une trainee basee sur la velocite"""
        speed = math.sqrt(vx*vx + vy*vy)
        count = max(1, int(speed / 2))
        direction = math.atan2(-vy, -vx)  # Direction opposee au mouvement
        self.emit(x, y, preset, count, direction, math.pi/4)

    def update(self, dt: float = 1.0):
        """Met a jour toutes les particules"""
        alive_particles = []

        for p in self.particles:
            p.life -= 1
            if p.life <= 0:
                continue

            # Physique
            p.x += p.vx * dt
            p.y += p.vy * dt

            # Gravite variable selon le type
            if p.type in ['smoke', 'magic']:
                p.vy -= p.gravity * 0.5  # Monte
            elif p.type in ['sparks', 'dust']:
                p.vy += p.gravity

            # Friction
            p.vx *= p.decay
            p.vy *= p.decay

            # Trail pour certains types
            if p.type in ['sparks', 'magic'] and p.life % 2 == 0:
                p.trail.append((p.x, p.y))
                if len(p.trail) > 5:
                    p.trail.pop(0)

            alive_particles.append(p)

        self.particles = alive_particles

    def draw(self, surface: pygame.Surface, camera_offset: Tuple[float, float] = (0, 0)):
        """Dessine les particules avec effets"""
        for p in self.particles:
            # Position avec camera
            px = int(p.x - camera_offset[0])
            py = int(p.y - camera_offset[1])

            # Calcul alpha basé sur la vie
            life_ratio = p.life / p.max_life
            alpha = int(255 * life_ratio)

            # Taille qui change avec la vie
            current_size = int(p.size * (0.5 + 0.5 * life_ratio))

            if current_size < 1:
                continue

            # Dessiner le trail
            if p.trail:
                for i, (tx, ty) in enumerate(p.trail):
                    trail_alpha = int(100 * (i / len(p.trail)) * life_ratio)
                    trail_size = max(1, current_size // 2)
                    trail_surface = pygame.Surface((trail_size*2, trail_size*2), pygame.SRCALPHA)
                    trail_color = (*p.color, trail_alpha)
                    pygame.draw.circle(trail_surface, trail_color, (trail_size, trail_size), trail_size)
                    surface.blit(trail_surface, (int(tx - trail_size - camera_offset[0]),
                                                 int(ty - trail_size - camera_offset[1])))

            # Glow effect
            if p.glow and current_size > 2:
                glow_size = current_size * 3
                glow_surface = pygame.Surface((glow_size*2, glow_size*2), pygame.SRCALPHA)
                glow_alpha = int(50 * life_ratio)
                glow_color = (*p.color, glow_alpha)
                pygame.draw.circle(glow_surface, glow_color, (glow_size, glow_size), glow_size)
                surface.blit(glow_surface, (px - glow_size, py - glow_size), special_flags=pygame.BLEND_ADD)

            # Particule principale
            particle_surface = pygame.Surface((current_size*2, current_size*2), pygame.SRCALPHA)
            particle_color = (*p.color, alpha)
            pygame.draw.circle(particle_surface, particle_color, (current_size, current_size), current_size)
            surface.blit(particle_surface, (px - current_size, py - current_size))


# ============================================================================
# FOND SPATIAL AVANCE
# ============================================================================

class SpaceBackground:
    """
    Fond spatial avec nebuleuses, etoiles et parallax
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.noise = OpenSimplex(seed=random.randint(0, 100000))

        # Couches d'etoiles pour parallax
        self.star_layers = []
        self._generate_stars()

        # Nebuleuse pre-rendue
        self.nebula_surface = None
        self._generate_nebula()

        # Planetes de fond
        self.background_planets = []
        self._generate_planets()

        # Animation
        self.time = 0

    def _generate_stars(self):
        """Genere plusieurs couches d'etoiles"""
        # 3 couches avec profondeurs differentes
        for layer_idx in range(3):
            stars = []
            count = 200 - layer_idx * 50  # Moins d'etoiles au premier plan
            size_range = (1 + layer_idx, 2 + layer_idx)
            parallax = 0.1 + layer_idx * 0.15  # Vitesse de parallax

            for _ in range(count):
                x = random.randint(0, self.width)
                y = random.randint(0, self.height)
                size = random.randint(*size_range)
                brightness = random.randint(150, 255)
                twinkle_speed = random.uniform(0.02, 0.08)
                twinkle_offset = random.uniform(0, 2 * math.pi)
                color_tint = random.choice([
                    (255, 255, 255),  # Blanc
                    (255, 220, 200),  # Chaud
                    (200, 220, 255),  # Froid
                    (255, 255, 200),  # Jaune
                ])

                stars.append({
                    'x': x, 'y': y, 'size': size, 'brightness': brightness,
                    'twinkle_speed': twinkle_speed, 'twinkle_offset': twinkle_offset,
                    'color': color_tint, 'parallax': parallax
                })

            self.star_layers.append(stars)

    def _generate_nebula(self):
        """Genere une texture de nebuleuse"""
        self.nebula_surface = pygame.Surface((self.width // 4, self.height // 4), pygame.SRCALPHA)

        # Couleurs de nebuleuse
        colors = [
            (80, 40, 120),   # Violet
            (120, 60, 80),   # Rose
            (40, 80, 120),   # Bleu
            (60, 40, 80),    # Pourpre fonce
        ]

        for x in range(0, self.width // 4, 2):
            for y in range(0, self.height // 4, 2):
                # Bruit multi-echelle
                noise_val = 0
                noise_val += self.noise.noise2(x * 0.01, y * 0.01) * 0.5
                noise_val += self.noise.noise2(x * 0.02, y * 0.02) * 0.3
                noise_val += self.noise.noise2(x * 0.04, y * 0.04) * 0.2

                noise_val = (noise_val + 1) / 2  # Normalise 0-1

                if noise_val > 0.4:  # Seuil pour creer des nuages
                    alpha = int((noise_val - 0.4) * 100)
                    color_idx = int(noise_val * (len(colors) - 1))
                    color = colors[min(color_idx, len(colors) - 1)]

                    pygame.draw.rect(
                        self.nebula_surface,
                        (*color, alpha),
                        (x, y, 2, 2)
                    )

        # Upscale avec filtre
        self.nebula_surface = pygame.transform.smoothscale(
            self.nebula_surface, (self.width, self.height)
        )

    def _generate_planets(self):
        """Genere des planetes de fond"""
        for _ in range(random.randint(1, 3)):
            self.background_planets.append({
                'x': random.randint(100, self.width - 100),
                'y': random.randint(50, self.height // 2),
                'radius': random.randint(30, 80),
                'color': random.choice([
                    (180, 100, 80),   # Mars-like
                    (200, 180, 160),  # Lune
                    (100, 150, 200),  # Neptune-like
                    (220, 200, 150),  # Venus-like
                ]),
                'parallax': random.uniform(0.05, 0.1)
            })

    def update(self, dt: float = 1.0):
        """Met a jour les animations"""
        self.time += dt * 0.016  # ~60fps base

    def draw(self, surface: pygame.Surface, camera_x: float = 0, camera_y: float = 0):
        """Dessine le fond avec parallax"""
        # Fond noir/bleu fonce
        surface.fill((5, 5, 15))

        # Nebuleuse (couche la plus lointaine)
        nebula_offset_x = int(camera_x * 0.02) % self.width
        nebula_offset_y = int(camera_y * 0.02) % self.height
        surface.blit(self.nebula_surface, (-nebula_offset_x, -nebula_offset_y))

        # Planetes de fond
        for planet in self.background_planets:
            px = int(planet['x'] - camera_x * planet['parallax']) % self.width
            py = int(planet['y'] - camera_y * planet['parallax'] * 0.5)

            # Ombre
            pygame.draw.circle(surface, (20, 20, 30), (px + 5, py + 5), planet['radius'])
            # Planete
            pygame.draw.circle(surface, planet['color'], (px, py), planet['radius'])
            # Highlight
            pygame.draw.circle(surface,
                             tuple(min(255, c + 50) for c in planet['color']),
                             (px - planet['radius']//3, py - planet['radius']//3),
                             planet['radius']//3)

        # Etoiles avec parallax et scintillement
        for layer in self.star_layers:
            for star in layer:
                # Position avec parallax
                sx = int(star['x'] - camera_x * star['parallax']) % self.width
                sy = int(star['y'] - camera_y * star['parallax'] * 0.5) % self.height

                # Scintillement
                twinkle = math.sin(self.time * star['twinkle_speed'] + star['twinkle_offset'])
                brightness = int(star['brightness'] * (0.7 + 0.3 * twinkle))

                # Couleur avec scintillement
                color = tuple(int(c * brightness / 255) for c in star['color'])

                if star['size'] <= 1:
                    surface.set_at((sx, sy), color)
                else:
                    pygame.draw.circle(surface, color, (sx, sy), star['size'])

                    # Halo pour les grosses etoiles
                    if star['size'] >= 3 and brightness > 200:
                        halo_surface = pygame.Surface((star['size']*6, star['size']*6), pygame.SRCALPHA)
                        halo_color = (*color, 30)
                        pygame.draw.circle(halo_surface, halo_color,
                                         (star['size']*3, star['size']*3), star['size']*3)
                        surface.blit(halo_surface,
                                   (sx - star['size']*3, sy - star['size']*3),
                                   special_flags=pygame.BLEND_ADD)


# ============================================================================
# VAISSEAU AVANCE
# ============================================================================

class AdvancedVesselRenderer:
    """
    Rendu avance du vaisseau spatial
    """

    def __init__(self):
        self.exhaust_particles = AdvancedParticleSystem(1000)
        self.damage_sparks = AdvancedParticleSystem(500)

    def draw(self, surface: pygame.Surface, x: float, y: float, angle: float,
             power: int, damage: float = 0, scale: float = 1.0):
        """
        Dessine le vaisseau avec effets
        damage: 0-1 (0 = intact, 1 = detruit)
        """
        # Corps principal
        size = int(20 * scale)

        # Points du vaisseau
        points = self._rotate_points([
            (0, -size),           # Nez
            (-size//2, size//2),  # Aile gauche
            (0, size//3),         # Base centre
            (size//2, size//2),   # Aile droite
        ], angle, x, y)

        # Couleur selon dommages
        base_color = (200, 200, 220)
        damage_color = (255, 100, 50)
        color = tuple(int(base_color[i] * (1 - damage) + damage_color[i] * damage)
                     for i in range(3))

        # Ombre
        shadow_points = [(p[0] + 3, p[1] + 3) for p in points]
        pygame.draw.polygon(surface, (20, 20, 30), shadow_points)

        # Corps
        pygame.draw.polygon(surface, color, points)
        pygame.draw.polygon(surface, (100, 100, 120), points, 2)

        # Cockpit
        cockpit_pos = self._rotate_point(0, -size//3, angle, x, y)
        pygame.draw.circle(surface, (100, 150, 200), cockpit_pos, size//4)
        pygame.draw.circle(surface, (150, 200, 255),
                          (cockpit_pos[0] - 2, cockpit_pos[1] - 2), size//6)

        # Flammes du reacteur
        if power > 0:
            self._draw_engine_flame(surface, x, y, angle, power, size, scale)

        # Etincelles de dommages
        if damage > 0.3 and random.random() < damage:
            spark_pos = self._rotate_point(
                random.uniform(-size//2, size//2),
                random.uniform(-size//2, size//2),
                angle, x, y
            )
            self.damage_sparks.emit(spark_pos[0], spark_pos[1], 'sparks', 3)

        # Mise a jour et dessin des particules
        self.exhaust_particles.update()
        self.exhaust_particles.draw(surface)
        self.damage_sparks.update()
        self.damage_sparks.draw(surface)

    def _draw_engine_flame(self, surface: pygame.Surface, x: float, y: float,
                           angle: float, power: int, size: int, scale: float):
        """Dessine la flamme du reacteur"""
        # Position de sortie du reacteur
        exhaust_pos = self._rotate_point(0, size//2 + 5, angle, x, y)

        # Direction de la flamme (opposee a l'angle)
        flame_angle = math.radians(angle + 180)

        # Longueur selon la puissance
        flame_length = (10 + power * 8) * scale

        # Plusieurs couches de flamme
        for i in range(3):
            layer_length = flame_length * (1 - i * 0.25)
            layer_width = (8 + power * 2 - i * 3) * scale

            # Variation aleatoire
            jitter = random.uniform(-3, 3)

            # Points de la flamme
            flame_points = [
                self._rotate_point(-layer_width/2, 0, angle, exhaust_pos[0], exhaust_pos[1]),
                self._rotate_point(layer_width/2, 0, angle, exhaust_pos[0], exhaust_pos[1]),
                self._rotate_point(jitter, layer_length, angle, exhaust_pos[0], exhaust_pos[1]),
            ]

            # Couleur de la couche
            colors = [
                (255, 255, 200),  # Coeur blanc
                (255, 200, 50),   # Milieu jaune
                (255, 100, 30),   # Exterieur orange
            ]

            pygame.draw.polygon(surface, colors[i], flame_points)

        # Particules d'echappement
        if random.random() < 0.3 + power * 0.15:
            self.exhaust_particles.emit(
                exhaust_pos[0], exhaust_pos[1],
                'flame',
                power + 1,
                math.radians(angle),
                math.pi / 4
            )

            # Fumee
            if power >= 3:
                smoke_pos = self._rotate_point(
                    random.uniform(-5, 5),
                    flame_length + random.uniform(0, 10),
                    angle, exhaust_pos[0], exhaust_pos[1]
                )
                self.exhaust_particles.emit(smoke_pos[0], smoke_pos[1], 'smoke', 2)

    def _rotate_point(self, px: float, py: float, angle: float,
                      cx: float, cy: float) -> Tuple[int, int]:
        """Rotation d'un point autour d'un centre"""
        rad = math.radians(-angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        rx = px * cos_a - py * sin_a + cx
        ry = px * sin_a + py * cos_a + cy

        return (int(rx), int(ry))

    def _rotate_points(self, points: List[Tuple[float, float]], angle: float,
                       cx: float, cy: float) -> List[Tuple[int, int]]:
        """Rotation de plusieurs points"""
        return [self._rotate_point(p[0], p[1], angle, cx, cy) for p in points]


# ============================================================================
# EFFETS POST-PROCESS
# ============================================================================

class PostProcessor:
    """
    Effets de post-traitement
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

        # Buffers pour effets
        self.glow_buffer = pygame.Surface((width // 4, height // 4))
        self.blur_buffer = pygame.Surface((width // 2, height // 2))

    def apply_bloom(self, surface: pygame.Surface, intensity: float = 0.3) -> pygame.Surface:
        """
        Applique un effet bloom (lueur des zones brillantes)
        """
        # Downscale
        small = pygame.transform.smoothscale(surface, (self.width // 4, self.height // 4))

        # Extraire les zones brillantes (simple threshold)
        # En pratique on ferait un vrai threshold mais ici on simule

        # Upscale avec flou implicite
        glow = pygame.transform.smoothscale(small, (self.width, self.height))

        # Blend additif
        result = surface.copy()
        glow.set_alpha(int(255 * intensity))
        result.blit(glow, (0, 0), special_flags=pygame.BLEND_ADD)

        return result

    def apply_vignette(self, surface: pygame.Surface, intensity: float = 0.4):
        """
        Applique un effet vignette (assombrissement des bords)
        """
        vignette = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        center_x, center_y = self.width // 2, self.height // 2
        max_dist = math.sqrt(center_x**2 + center_y**2)

        # Dessiner des cercles concentriques
        for i in range(10):
            radius = int(max_dist * (1 - i * 0.1))
            alpha = int(intensity * 255 * (i / 10))
            pygame.draw.circle(vignette, (0, 0, 0, alpha), (center_x, center_y), radius)

        surface.blit(vignette, (0, 0))

    def apply_scanlines(self, surface: pygame.Surface, intensity: float = 0.1):
        """
        Applique des lignes de scan CRT
        """
        for y in range(0, self.height, 3):
            pygame.draw.line(surface, (0, 0, 0), (0, y), (self.width, y))


# ============================================================================
# CAMERA AVANCEE
# ============================================================================

class SmoothCamera:
    """
    Camera avec mouvements fluides et effets
    """

    def __init__(self, width: int, height: int, world_width: int, world_height: int):
        self.width = width
        self.height = height
        self.world_width = world_width
        self.world_height = world_height

        self.x = 0
        self.y = 0
        self.target_x = 0
        self.target_y = 0

        self.zoom = 1.0
        self.target_zoom = 1.0

        self.shake_intensity = 0
        self.shake_decay = 0.9

        # Smoothing
        self.smoothing = 0.1

    def follow(self, target_x: float, target_y: float, instant: bool = False):
        """Suit une cible"""
        self.target_x = target_x - self.width // 2
        self.target_y = target_y - self.height // 2

        if instant:
            self.x = self.target_x
            self.y = self.target_y

    def set_zoom(self, zoom: float, instant: bool = False):
        """Definit le zoom"""
        self.target_zoom = max(0.5, min(2.0, zoom))
        if instant:
            self.zoom = self.target_zoom

    def shake(self, intensity: float):
        """Declenche un screen shake"""
        self.shake_intensity = intensity

    def update(self, dt: float = 1.0):
        """Met a jour la camera"""
        # Interpolation douce
        self.x += (self.target_x - self.x) * self.smoothing * dt
        self.y += (self.target_y - self.y) * self.smoothing * dt
        self.zoom += (self.target_zoom - self.zoom) * self.smoothing * dt

        # Limites
        self.x = max(0, min(self.world_width - self.width, self.x))
        self.y = max(0, min(self.world_height - self.height, self.y))

        # Decay du shake
        self.shake_intensity *= self.shake_decay

    def get_offset(self) -> Tuple[float, float]:
        """Retourne l'offset avec shake"""
        shake_x = random.uniform(-self.shake_intensity, self.shake_intensity)
        shake_y = random.uniform(-self.shake_intensity, self.shake_intensity)
        return (self.x + shake_x, self.y + shake_y)

    def world_to_screen(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """Convertit coordonnees monde en ecran"""
        offset = self.get_offset()
        screen_x = int((world_x - offset[0]) * self.zoom)
        screen_y = int((world_y - offset[1]) * self.zoom)
        return (screen_x, screen_y)


# Test du module
if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((1200, 800))
    pygame.display.set_caption("Pro Graphics Test")
    clock = pygame.time.Clock()

    # Test des composants
    background = SpaceBackground(1200, 800)
    particles = AdvancedParticleSystem()
    vessel = AdvancedVesselRenderer()
    terrain = ProceduralTerrain(1200)
    terrain_points, zones = terrain.generate()

    running = True
    angle = 0
    power = 2
    camera_x = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                particles.emit(event.pos[0], event.pos[1], 'explosion', 50)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            angle -= 2
        if keys[pygame.K_RIGHT]:
            angle += 2
        if keys[pygame.K_UP]:
            power = min(4, power + 1)
        if keys[pygame.K_DOWN]:
            power = max(0, power - 1)

        camera_x += 1

        # Mise a jour
        background.update()
        particles.update()

        # Rendu
        background.draw(screen, camera_x, 0)

        # Terrain
        if terrain_points:
            shifted_points = [(p[0] - camera_x % 1200, 800 - p[1]) for p in terrain_points]
            pygame.draw.lines(screen, (150, 100, 80), False, shifted_points, 3)

        # Vaisseau
        vessel.draw(screen, 600, 400, angle, power, damage=0.2)

        # Particules
        particles.draw(screen)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
