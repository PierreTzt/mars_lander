"""
Module d'affichage avancé pour Mars Lander - Style Réaliste/Spatial.

Ce module gère tout l'aspect visuel du jeu avec un rendu immersif:
- Fond étoilé dynamique avec scintillement
- Arrière-plan martien avec dégradé et montagnes
- Système de particules pour flammes, explosions et poussière
- HUD (Head-Up Display) style science-fiction
- Dessin procédural de la fusée

ARCHITECTURE GRAPHIQUE:
======================
Le rendu se fait en couches (de l'arrière vers l'avant):
1. Dégradé de ciel martien (MarsBackground)
2. Champ d'étoiles scintillantes (StarField)
3. Terrain de Mars avec zone d'atterrissage
4. Vaisseau spatial avec effets de propulsion
5. Particules (flammes, explosions, poussière)
6. Interface utilisateur HUD (SciFiHUD)

SYSTÈME DE PARTICULES:
=====================
Les particules simulent des effets visuels réalistes:
- Flammes de propulsion: jaillissent du moteur
- Explosions: se dispersent dans toutes les directions
- Poussière: s'élève lors de l'atterrissage

Chaque particule a:
- Position (x, y) et vitesse (vx, vy)
- Couleur avec fondu progressif
- Durée de vie limitée
"""

import pygame
import math
import random
from typing import Dict, Tuple, List, Optional

from data import (
    fenX, fenY, echelle, img_par_sec, ia_active,
    max_h_speed, max_v_speed,
    trajectoire_active, trajectoire_points,
    graphique_actif, graphique_points,
    vent_actif, gravite,
    # Couleurs de base
    BLANC, NOIR, ROUGE, VERT, ORANGE, JAUNE,
    # Couleurs spatiales
    NOIR_ESPACE, ETOILE_DIM, ETOILE_MEDIUM, ETOILE_BRIGHT, ETOILE_BLEU, ETOILE_ROUGE,
    # Mars
    MARS_CIEL_HAUT, MARS_CIEL_MILIEU, MARS_CIEL_BAS,
    MARS_SOL, MARS_SOL_CLAIR, MARS_SOL_SOMBRE, MARS_ROCHE, MARS_OMBRE,
    # Zone d'atterrissage
    ZONE_ATTERRISSAGE, ZONE_ATTERRISSAGE_DIM, BALISE_LUMIERE,
    # Flammes
    FLAMME_COEUR, FLAMME_INTERIEUR, FLAMME_MILIEU, FLAMME_EXTERIEUR,
    # Effets
    EXPLOSION_JAUNE, EXPLOSION_ORANGE, EXPLOSION_ROUGE,
    POUSSIERE_CLAIRE, POUSSIERE_SOMBRE,
    SUCCES_VERT, CRASH_ROUGE,
    # HUD
    HUD_PRIMAIRE, HUD_SECONDAIRE, HUD_ACCENT, HUD_ALERTE, HUD_DANGER,
    HUD_FOND, HUD_BORDURE, HUD_TEXTE,
    # Trajectoire
    TRAJECTOIRE, TRAJECTOIRE_DANGER
)


# =============================================================================
# SYSTÈME DE PARTICULES
# =============================================================================
# Les particules créent des effets visuels dynamiques comme les flammes
# de propulsion, les explosions et la poussière. Chaque particule est
# un petit élément graphique avec sa propre physique simplifiée.
# =============================================================================

class Particle:
    """
    Représente une particule individuelle dans le système de particules.

    Une particule est un petit élément visuel avec:
    - Une position qui évolue selon sa vitesse
    - Une durée de vie limitée (disparaît progressivement)
    - Une couleur et une taille qui diminuent avec le temps

    CYCLE DE VIE:
    1. Création avec position, vitesse, couleur, taille, durée
    2. À chaque frame: mise à jour position, réduction durée de vie
    3. Fondu progressif (alpha diminue)
    4. Mort quand lifetime atteint 0

    Attributes:
        x, y (float): Position actuelle de la particule
        vx, vy (float): Vitesse en pixels par frame
        color (Tuple): Couleur RGB ou RGBA
        size (float): Taille initiale en pixels
        lifetime (int): Frames restantes avant disparition
        max_lifetime (int): Durée totale pour calculer le fondu
        alive (bool): False quand la particule doit être supprimée
    """

    def __init__(self, x: float, y: float, vx: float, vy: float,
                 color: Tuple, size: float, lifetime: int):
        """
        Crée une nouvelle particule.

        Args:
            x, y: Position initiale
            vx, vy: Vitesse initiale (pixels/frame)
            color: Couleur RGB ou RGBA
            size: Taille en pixels
            lifetime: Durée de vie en frames
        """
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.size = size
        self.lifetime = lifetime
        self.max_lifetime = lifetime  # Mémorisé pour calculer le ratio de fondu
        self.alive = True

    def update(self) -> None:
        """
        Met à jour la particule pour une frame.

        Déplace la particule selon sa vitesse et décrémente
        sa durée de vie. Quand lifetime atteint 0, la particule
        est marquée comme morte pour suppression.
        """
        # Déplacement selon la vitesse
        self.x += self.vx
        self.y += self.vy

        # Réduction de la durée de vie
        self.lifetime -= 1
        if self.lifetime <= 0:
            self.alive = False

    def draw(self, surface: pygame.Surface) -> None:
        """
        Dessine la particule sur une surface Pygame.

        Applique un effet de fondu: plus la particule vieillit,
        plus elle devient transparente et petite.

        Args:
            surface: Surface Pygame sur laquelle dessiner
        """
        if not self.alive:
            return

        # === CALCUL DU FONDU (FADE OUT) ===
        # Alpha proportionnel à la durée de vie restante
        # Particule neuve = 255 (opaque), mourante = 0 (transparent)
        alpha = int(255 * (self.lifetime / self.max_lifetime))

        # Taille diminue aussi avec le temps (effet de dissipation)
        current_size = max(1, int(self.size * (self.lifetime / self.max_lifetime)))

        # === GESTION DE LA COULEUR AVEC ALPHA ===
        if len(self.color) == 3:
            # Couleur RGB -> ajouter alpha
            color_with_alpha = (*self.color, alpha)
        else:
            # Couleur RGBA -> prendre le minimum des alphas
            color_with_alpha = (*self.color[:3], min(alpha, self.color[3]))

        # === DESSIN DE LA PARTICULE ===
        if current_size > 1:
            # Cercle pour les particules visibles
            pygame.draw.circle(surface, self.color[:3],
                             (int(self.x), int(self.y)), current_size)
        else:
            # Simple pixel pour les petites particules
            surface.set_at((int(self.x), int(self.y)), self.color[:3])


class ParticleSystem:
    """
    Gestionnaire du système de particules.

    Cette classe centralise la création, mise à jour et rendu
    de toutes les particules du jeu. Elle fournit des méthodes
    spécialisées pour différents types d'effets:

    TYPES D'EFFETS:
    - Propulsion (emit_thrust): Flammes sortant du moteur
    - Explosion (emit_explosion): Dispersion lors d'un crash
    - Poussière (emit_dust): Nuage lors de l'atterrissage

    OPTIMISATION:
    Les particules mortes sont automatiquement supprimées
    à chaque update pour éviter l'accumulation en mémoire.

    Attributes:
        particles (List[Particle]): Liste de toutes les particules actives
    """

    def __init__(self):
        """Initialise le système avec une liste vide de particules."""
        self.particles: List[Particle] = []

    def add_particle(self, x: float, y: float, vx: float, vy: float,
                     color: Tuple, size: float, lifetime: int) -> None:
        """
        Ajoute une particule au système.

        Args:
            x, y: Position de spawn
            vx, vy: Vitesse initiale
            color: Couleur de la particule
            size: Taille en pixels
            lifetime: Durée de vie en frames
        """
        self.particles.append(Particle(x, y, vx, vy, color, size, lifetime))

    def emit_thrust(self, x: float, y: float, angle: float, power: int) -> None:
        """
        Émet des particules de propulsion (flammes du moteur).

        Les particules sont éjectées dans la direction OPPOSÉE
        à l'orientation du vaisseau (le moteur pousse vers le bas,
        les flammes vont vers le bas aussi).

        CALCUL DE DIRECTION:
        - sin(angle) donne la composante horizontale
        - cos(angle) donne la composante verticale
        - Les flammes vont dans la même direction (poussée réactive)

        Args:
            x, y: Position de la tuyère du moteur
            angle: Angle du vaisseau en degrés
            power: Puissance du moteur (0-4), détermine l'intensité
        """
        # Pas de flamme si moteur éteint
        if power <= 0:
            return

        # Convertir l'angle en radians pour les calculs trigonométriques
        angle_rad = math.radians(angle)
        sin_a = math.sin(angle_rad)
        cos_a = math.cos(angle_rad)

        # Créer plusieurs particules (plus de puissance = plus de particules)
        for _ in range(power * 2):
            # Direction de la flamme avec légère dispersion aléatoire
            spread = random.uniform(-0.3, 0.3)  # Angle de dispersion
            speed = random.uniform(2, 5) * power  # Vitesse proportionnelle à la puissance

            # Calcul des composantes de vitesse
            vx = sin_a * speed + random.uniform(-0.5, 0.5)
            vy = cos_a * speed + random.uniform(-0.5, 0.5)

            # Couleur aléatoire parmi les teintes de flamme
            # Du coeur bleu/blanc au rouge extérieur
            colors = [FLAMME_COEUR, FLAMME_INTERIEUR, FLAMME_MILIEU, FLAMME_EXTERIEUR]
            color = random.choice(colors)

            # Taille et durée de vie aléatoires
            size = random.uniform(2, 4 + power)
            lifetime = random.randint(10, 25)

            self.add_particle(x, y, vx, vy, color, size, lifetime)

    def emit_explosion(self, x: float, y: float, intensity: int = 30) -> None:
        """
        Émet des particules d'explosion lors d'un crash.

        Les particules sont projetées dans toutes les directions
        depuis le point d'impact, simulant une explosion.

        Args:
            x, y: Centre de l'explosion (position du crash)
            intensity: Nombre de particules à créer
        """
        for _ in range(intensity):
            # Direction aléatoire sur 360°
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 8)  # Vitesse variable

            # Composantes de vitesse
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed - random.uniform(0, 2)  # Légère tendance vers le haut

            # Couleurs d'explosion (jaune -> orange -> rouge)
            colors = [EXPLOSION_JAUNE, EXPLOSION_ORANGE, EXPLOSION_ROUGE]
            color = random.choice(colors)

            size = random.uniform(3, 8)  # Particules plus grosses que les flammes
            lifetime = random.randint(20, 50)  # Durée plus longue

            self.add_particle(x, y, vx, vy, color, size, lifetime)

    def emit_dust(self, x: float, y: float, intensity: int = 20) -> None:
        """
        Émet des particules de poussière lors de l'atterrissage.

        La poussière s'élève et se disperse horizontalement,
        simulant l'impact du vaisseau sur le sol martien.

        Args:
            x, y: Point d'impact (base du vaisseau)
            intensity: Nombre de particules
        """
        for _ in range(intensity):
            # Direction vers le haut uniquement (arc de -π à 0)
            angle = random.uniform(-math.pi, 0)
            speed = random.uniform(1, 4)

            # Plus de dispersion horizontale que verticale
            vx = math.cos(angle) * speed * 2
            vy = math.sin(angle) * speed

            # Couleurs de poussière martienne (tons bruns/rouges)
            colors = [POUSSIERE_CLAIRE, POUSSIERE_SOMBRE]
            color = random.choice(colors)

            size = random.uniform(2, 5)
            lifetime = random.randint(30, 60)  # Poussière flotte plus longtemps

            self.add_particle(x, y, vx, vy, color, size, lifetime)

    def update(self) -> None:
        """
        Met à jour toutes les particules du système.

        Pour chaque particule:
        1. Mise à jour position et durée de vie
        2. Application d'une légère gravité
        3. Suppression si morte
        """
        for particle in self.particles:
            particle.update()
            # Gravité légère pour effet réaliste (les particules retombent)
            particle.vy += 0.05

        # Nettoyage: supprimer les particules mortes
        # Utilise une list comprehension pour l'efficacité
        self.particles = [p for p in self.particles if p.alive]

    def draw(self, surface: pygame.Surface) -> None:
        """
        Dessine toutes les particules sur la surface.

        Args:
            surface: Surface Pygame cible
        """
        for particle in self.particles:
            particle.draw(surface)


# =============================================================================
# FOND ÉTOILÉ
# =============================================================================
# Le champ d'étoiles crée une atmosphère spatiale immersive.
# Les étoiles ont différentes tailles, luminosités et couleurs.
# Un effet de scintillement les rend vivantes.
# =============================================================================

class StarField:
    """
    Champ d'étoiles avec effet de scintillement.

    Génère un fond étoilé réaliste avec:
    - Étoiles de différentes tailles (1-3 pixels)
    - Luminosités variables (0.3 à 1.0)
    - Couleurs variées (blanc, bleu, rouge)
    - Animation de scintillement (twinkle)

    SCINTILLEMENT:
    Chaque étoile a sa propre phase et vitesse de scintillement,
    créé par une fonction sinusoïdale. Cela simule l'effet
    atmosphérique que l'on voit depuis la Terre.

    Attributes:
        width, height (int): Dimensions de la zone d'étoiles
        stars (List[Dict]): Liste des étoiles avec leurs propriétés
        frame (int): Compteur de frames pour l'animation
    """

    def __init__(self, width: int, height: int, num_stars: int = 200):
        """
        Génère le champ d'étoiles.

        Args:
            width, height: Dimensions de la zone
            num_stars: Nombre d'étoiles à générer
        """
        self.width = width
        self.height = height
        self.stars: List[Dict] = []

        # Génération procédurale des étoiles
        for _ in range(num_stars):
            self.stars.append({
                # Position: uniquement dans la partie haute (ciel)
                'x': random.randint(0, width),
                'y': random.randint(0, int(height * 0.7)),

                # Apparence
                'size': random.choice([1, 1, 1, 2, 2, 3]),  # Plus de petites étoiles
                'brightness': random.uniform(0.3, 1.0),  # Luminosité de base

                # Animation de scintillement
                'twinkle_speed': random.uniform(0.02, 0.08),  # Vitesse variable
                'twinkle_offset': random.uniform(0, 2 * math.pi),  # Phase initiale

                # Couleur (la plupart blanches, quelques colorées)
                'color': random.choice([ETOILE_DIM, ETOILE_MEDIUM, ETOILE_BRIGHT,
                                       ETOILE_BLEU, ETOILE_ROUGE])
            })

        self.frame = 0  # Compteur pour l'animation

    def update(self) -> None:
        """Incrémente le compteur de frames pour l'animation."""
        self.frame += 1

    def draw(self, surface: pygame.Surface) -> None:
        """
        Dessine toutes les étoiles avec effet de scintillement.

        Le scintillement est calculé avec:
        brightness = base * (0.7 + 0.3 * sin(frame * speed + offset))

        Cela fait varier la luminosité entre 70% et 100% de la valeur de base.

        Args:
            surface: Surface Pygame cible
        """
        for star in self.stars:
            # === CALCUL DU SCINTILLEMENT ===
            # sin() retourne une valeur entre -1 et 1
            # On la transforme en valeur entre 0.7 et 1.0
            twinkle = math.sin(self.frame * star['twinkle_speed'] + star['twinkle_offset'])
            brightness = star['brightness'] * (0.7 + 0.3 * twinkle)

            # Appliquer la luminosité à la couleur
            color = tuple(int(c * brightness) for c in star['color'][:3])

            # === DESSIN DE L'ÉTOILE ===
            if star['size'] == 1:
                # Simple pixel pour les petites étoiles (plus rapide)
                surface.set_at((star['x'], star['y']), color)
            else:
                # Cercle pour les plus grandes
                pygame.draw.circle(surface, color,
                                 (star['x'], star['y']), star['size'])

                # Halo lumineux pour les très grosses étoiles brillantes
                if star['size'] >= 3 and brightness > 0.8:
                    halo_color = tuple(int(c * 0.3) for c in color)
                    pygame.draw.circle(surface, halo_color,
                                     (star['x'], star['y']), star['size'] + 2)


# =============================================================================
# ARRIÈRE-PLAN MARTIEN
# =============================================================================
# Crée l'atmosphère martienne avec un dégradé de ciel
# et des silhouettes de montagnes en arrière-plan.
# =============================================================================

class MarsBackground:
    """
    Arrière-plan avec dégradé de ciel martien et montagnes.

    Génère une vue atmosphérique de Mars:
    - Dégradé vertical du noir spatial au orange/brun de l'horizon
    - Deux couches de montagnes (lointaines et proches)
    - Surface pré-rendue pour performance

    STRUCTURE DU CIEL:
    - 0-30% hauteur: Noir spatial → Violet sombre
    - 30-70% hauteur: Violet → Orange/brun
    - 70-100%: Orange de l'horizon

    Attributes:
        width, height (int): Dimensions de l'arrière-plan
        background_surface (Surface): Surface pré-rendue (cache)
        mountains (List): Points des silhouettes de montagnes
    """

    def __init__(self, width: int, height: int):
        """
        Initialise l'arrière-plan.

        Args:
            width, height: Dimensions de la fenêtre
        """
        self.width = width
        self.height = height
        self.background_surface = None  # Sera créé au premier rendu
        self.mountains = self._generate_mountains()

    def _generate_mountains(self) -> List[List[Tuple[int, int]]]:
        """
        Génère procéduralement les silhouettes de montagnes.

        Crée deux couches de montagnes:
        - Lointaines: plus hautes dans le ciel, couleur plus claire
        - Proches: plus basses, couleur plus sombre

        ALGORITHME:
        Pour chaque couche, on génère des points avec:
        - X croissant par pas aléatoire (80-150 ou 50-100 pixels)
        - Y aléatoire pour créer les pics

        Returns:
            Liste de tuples (type, points) pour chaque couche
        """
        mountains = []

        # === MONTAGNES LOINTAINES ===
        # Plus hautes (50-70% de la hauteur), espacement large
        mountain1 = [(0, self.height)]  # Point de départ en bas à gauche
        x = 0
        while x < self.width:
            # Hauteur du pic (plus haut = Y plus petit)
            peak_height = random.randint(int(self.height * 0.5), int(self.height * 0.7))
            mountain1.append((x, peak_height))
            x += random.randint(80, 150)  # Espacement large
        mountain1.append((self.width, self.height))  # Fermer le polygone
        mountains.append(('far', mountain1))

        # === MONTAGNES PROCHES ===
        # Plus basses (60-80% de la hauteur), espacement serré
        mountain2 = [(0, self.height)]
        x = 0
        while x < self.width:
            peak_height = random.randint(int(self.height * 0.6), int(self.height * 0.8))
            mountain2.append((x, peak_height))
            x += random.randint(50, 100)  # Espacement plus serré
        mountain2.append((self.width, self.height))
        mountains.append(('near', mountain2))

        return mountains

    def _draw_gradient(self, surface: pygame.Surface) -> None:
        """
        Dessine le dégradé de ciel martien.

        Utilise une interpolation linéaire (lerp) entre trois couleurs:
        - MARS_CIEL_HAUT: Noir/violet spatial (haut)
        - MARS_CIEL_MILIEU: Transition violet/orange
        - MARS_CIEL_BAS: Orange/brun de l'horizon

        Args:
            surface: Surface sur laquelle dessiner
        """
        for y in range(self.height):
            # Déterminer dans quelle zone on se trouve et interpoler
            if y < self.height * 0.3:
                # Zone haute: noir → violet
                t = y / (self.height * 0.3)
                color = self._lerp_color(MARS_CIEL_HAUT, MARS_CIEL_MILIEU, t)
            elif y < self.height * 0.7:
                # Zone médiane: violet → orange
                t = (y - self.height * 0.3) / (self.height * 0.4)
                color = self._lerp_color(MARS_CIEL_MILIEU, MARS_CIEL_BAS, t)
            else:
                # Zone basse: orange constant
                color = MARS_CIEL_BAS

            # Dessiner une ligne horizontale de cette couleur
            pygame.draw.line(surface, color, (0, y), (self.width, y))

    def _lerp_color(self, c1: Tuple, c2: Tuple, t: float) -> Tuple:
        """
        Interpolation linéaire entre deux couleurs.

        Formule: result = c1 + (c2 - c1) * t

        Args:
            c1: Couleur de départ (RGB)
            c2: Couleur d'arrivée (RGB)
            t: Facteur d'interpolation (0.0 = c1, 1.0 = c2)

        Returns:
            Couleur interpolée (RGB)
        """
        return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

    def init_background(self) -> None:
        """
        Initialise la surface d'arrière-plan (appelé une seule fois).

        Pré-rend le dégradé et les montagnes sur une surface
        qui sera blittée à chaque frame. C'est une optimisation
        car ces éléments ne changent jamais.
        """
        self.background_surface = pygame.Surface((self.width, self.height))

        # 1. Dessiner le dégradé de ciel
        self._draw_gradient(self.background_surface)

        # 2. Dessiner les montagnes (par-dessus le dégradé)
        for layer, points in self.mountains:
            if layer == 'far':
                # Montagnes lointaines: couleur plus claire (atmosphère)
                color = MARS_SOL_SOMBRE
            else:
                # Montagnes proches: couleur plus sombre
                color = MARS_ROCHE
            pygame.draw.polygon(self.background_surface, color, points)

    def draw(self, surface: pygame.Surface) -> None:
        """
        Dessine l'arrière-plan sur la surface de jeu.

        Initialise la surface au premier appel, puis utilise
        le cache pour les appels suivants.

        Args:
            surface: Surface Pygame cible
        """
        if self.background_surface is None:
            self.init_background()
        surface.blit(self.background_surface, (0, 0))


# =============================================================================
# HUD SCI-FI (HEAD-UP DISPLAY)
# =============================================================================
# Interface utilisateur style science-fiction avec panneaux,
# barres de progression, valeurs et alertes.
# =============================================================================

class SciFiHUD:
    """
    Interface utilisateur style science-fiction.

    Affiche les informations de jeu dans un style futuriste:
    - Panneaux semi-transparents avec bordures
    - Barres de progression avec dégradés
    - Valeurs avec labels colorés
    - Alertes clignotantes
    - Mini-radar avec animation de balayage

    DESIGN:
    - Couleurs bleu/cyan pour l'information normale
    - Orange pour les alertes
    - Rouge pour les dangers
    - Coins stylisés sur les panneaux

    Attributes:
        width, height (int): Dimensions de l'écran
        font_large, font_medium, font_small: Polices Pygame
        frame (int): Compteur pour les animations
    """

    def __init__(self, width: int, height: int):
        """
        Initialise le HUD.

        Args:
            width, height: Dimensions de l'écran
        """
        self.width = width
        self.height = height
        self.font_large = None
        self.font_medium = None
        self.font_small = None
        self.frame = 0  # Compteur pour animations

    def _init_fonts(self) -> None:
        """
        Initialise les polices (appelé paresseusement).

        Les polices sont créées au premier besoin car Pygame
        doit être initialisé avant de créer des Font.
        """
        if self.font_large is None:
            self.font_large = pygame.font.Font(None, 32)
            self.font_medium = pygame.font.Font(None, 24)
            self.font_small = pygame.font.Font(None, 18)

    def draw_panel(self, surface: pygame.Surface, x: int, y: int,
                   width: int, height: int, title: str = "") -> None:
        """
        Dessine un panneau HUD stylisé.

        Le panneau comprend:
        - Fond semi-transparent
        - Bordure complète
        - Coins accentués (lignes de 10px)
        - Titre optionnel avec ligne de séparation

        Args:
            surface: Surface cible
            x, y: Position du coin supérieur gauche
            width, height: Dimensions du panneau
            title: Titre optionnel affiché en haut
        """
        # === FOND SEMI-TRANSPARENT ===
        panel = pygame.Surface((width, height), pygame.SRCALPHA)
        panel.fill((*HUD_FOND, 180))  # Alpha = 180/255
        surface.blit(panel, (x, y))

        # === BORDURE ===
        pygame.draw.rect(surface, HUD_BORDURE, (x, y, width, height), 2)

        # === COINS ACCENTUÉS ===
        # Petites lignes aux 4 coins pour un look futuriste
        corner_size = 10

        # Coin haut gauche
        pygame.draw.line(surface, HUD_PRIMAIRE, (x, y + corner_size), (x, y), 2)
        pygame.draw.line(surface, HUD_PRIMAIRE, (x, y), (x + corner_size, y), 2)

        # Coin haut droit
        pygame.draw.line(surface, HUD_PRIMAIRE, (x + width - corner_size, y), (x + width, y), 2)
        pygame.draw.line(surface, HUD_PRIMAIRE, (x + width, y), (x + width, y + corner_size), 2)

        # Coin bas gauche
        pygame.draw.line(surface, HUD_PRIMAIRE, (x, y + height - corner_size), (x, y + height), 2)
        pygame.draw.line(surface, HUD_PRIMAIRE, (x, y + height), (x + corner_size, y + height), 2)

        # Coin bas droit
        pygame.draw.line(surface, HUD_PRIMAIRE, (x + width - corner_size, y + height), (x + width, y + height), 2)
        pygame.draw.line(surface, HUD_PRIMAIRE, (x + width, y + height), (x + width, y + height - corner_size), 2)

        # === TITRE ===
        if title:
            title_text = self.font_medium.render(title, True, HUD_ACCENT)
            surface.blit(title_text, (x + 10, y + 5))
            # Ligne de séparation sous le titre
            pygame.draw.line(surface, HUD_SECONDAIRE,
                           (x + 5, y + 25), (x + width - 5, y + 25), 1)

    def draw_progress_bar(self, surface: pygame.Surface, x: int, y: int,
                          width: int, height: int, value: float, max_value: float,
                          color: Tuple = HUD_PRIMAIRE, show_text: bool = True) -> None:
        """
        Dessine une barre de progression avec dégradé.

        La barre se remplit de gauche à droite proportionnellement
        à value/max_value. Un dégradé horizontal donne du relief.

        Args:
            surface: Surface cible
            x, y: Position
            width, height: Dimensions
            value: Valeur actuelle
            max_value: Valeur maximale
            color: Couleur de remplissage
            show_text: Non utilisé (conservé pour compatibilité)
        """
        # === FOND DE LA BARRE ===
        pygame.draw.rect(surface, HUD_FOND, (x, y, width, height))
        pygame.draw.rect(surface, HUD_BORDURE, (x, y, width, height), 1)

        # === CALCUL DU REMPLISSAGE ===
        if max_value > 0:
            fill_width = int((value / max_value) * (width - 4))
            fill_width = max(0, min(fill_width, width - 4))  # Borner la valeur

            if fill_width > 0:
                # === DÉGRADÉ HORIZONTAL ===
                # Plus clair au centre, plus sombre aux bords
                for i in range(fill_width):
                    t = i / max(1, fill_width)  # Position relative 0-1
                    # Luminosité varie de 50% à 100%
                    segment_color = tuple(int(color[j] * (0.5 + 0.5 * t)) for j in range(3))
                    pygame.draw.line(surface, segment_color,
                                   (x + 2 + i, y + 2), (x + 2 + i, y + height - 2))

    def draw_value(self, surface: pygame.Surface, x: int, y: int,
                   label: str, value: any, color: Tuple = HUD_TEXTE) -> None:
        """
        Affiche une valeur avec son label.

        Format: "Label: Valeur"
        Le label est en couleur secondaire, la valeur en couleur spécifiée.

        Args:
            surface: Surface cible
            x, y: Position du texte
            label: Nom de la valeur (ex: "Vitesse H")
            value: Valeur à afficher
            color: Couleur de la valeur (peut indiquer un danger)
        """
        self._init_fonts()

        # Label en couleur secondaire
        label_text = self.font_small.render(f"{label}:", True, HUD_SECONDAIRE)
        surface.blit(label_text, (x, y))

        # Formatage de la valeur
        if isinstance(value, float):
            value_str = f"{value:.1f}"  # 1 décimale pour les floats
        else:
            value_str = str(value)

        # Valeur en couleur spécifiée
        value_text = self.font_small.render(value_str, True, color)
        surface.blit(value_text, (x + 80, y))

    def draw_alert(self, surface: pygame.Surface, x: int, y: int,
                   message: str, level: str = "warning") -> None:
        """
        Affiche une alerte avec effet de clignotement.

        Les alertes "danger" clignotent en alternant blanc/rouge.
        Les alertes "warning" restent en orange fixe.

        Args:
            surface: Surface cible
            x, y: Position de l'alerte
            message: Texte de l'alerte
            level: "warning" ou "danger"
        """
        self._init_fonts()

        if level == "danger":
            color = HUD_DANGER
            # Effet de clignotement: alterne toutes les 10 frames
            if (self.frame // 10) % 2 == 0:
                color = BLANC
        else:
            color = HUD_ALERTE

        # Symbole d'alerte + message
        text = self.font_medium.render(f"⚠ {message}", True, color)
        surface.blit(text, (x, y))

    def draw_mini_radar(self, surface: pygame.Surface, x: int, y: int,
                        size: int, ship_pos: Tuple, zone_pos: Tuple,
                        zone_width: float) -> None:
        """
        Dessine un mini-radar montrant la position relative.

        Le radar affiche:
        - Cercles concentriques (fond)
        - Ligne de balayage rotative (animation)
        - Zone d'atterrissage (ligne verte en bas)
        - Position du vaisseau (point cyan)

        Args:
            surface: Surface cible
            x, y: Position du radar
            size: Diamètre du radar
            ship_pos: Position (x, y) du vaisseau
            zone_pos: Centre de la zone d'atterrissage
            zone_width: Largeur de la zone
        """
        center_x = x + size // 2
        center_y = y + size // 2
        radius = size // 2

        # === FOND DU RADAR ===
        pygame.draw.circle(surface, HUD_FOND, (center_x, center_y), radius)
        pygame.draw.circle(surface, HUD_BORDURE, (center_x, center_y), radius, 1)

        # === CERCLES CONCENTRIQUES ===
        for r in range(size // 6, radius, size // 6):
            pygame.draw.circle(surface, HUD_SECONDAIRE,
                             (center_x, center_y), r, 1)

        # === LIGNE DE BALAYAGE ROTATIVE ===
        # Tourne de 2° par frame
        angle = (self.frame * 2) % 360
        end_x = center_x + int(math.cos(math.radians(angle)) * (radius - 5))
        end_y = center_y + int(math.sin(math.radians(angle)) * (radius - 5))
        pygame.draw.line(surface, HUD_ACCENT,
                        (center_x, center_y), (end_x, end_y), 1)

        # === ZONE D'ATTERRISSAGE (ligne verte) ===
        pygame.draw.line(surface, ZONE_ATTERRISSAGE,
                        (center_x - 10, y + size - 15),
                        (center_x + 10, y + size - 15), 3)

        # === POSITION DU VAISSEAU ===
        # Calcul de la position relative au centre de la zone
        ship_rel_x = (ship_pos[0] - zone_pos[0]) / max(1, zone_width) * (radius)
        ship_rel_x = max(-radius + 5, min(radius - 5, ship_rel_x))
        ship_screen_x = center_x + int(ship_rel_x * 0.3)
        ship_screen_y = y + size // 4 + int(ship_pos[1] / 3000 * (radius))

        pygame.draw.circle(surface, HUD_ACCENT, (ship_screen_x, ship_screen_y), 3)

    def update(self) -> None:
        """Incrémente le compteur de frames pour les animations."""
        self.frame += 1


# =============================================================================
# AFFICHAGE PRINCIPAL
# =============================================================================
# Classe principale orchestrant tout le rendu du jeu.
# Coordonne tous les sous-systèmes graphiques.
# =============================================================================

class Affichage:
    """
    Gestionnaire principal de l'affichage du jeu Mars Lander.

    Cette classe coordonne tous les éléments visuels:
    - Initialisation de Pygame et de la fenêtre
    - Gestion des sous-systèmes (étoiles, fond, particules, HUD)
    - Dessin du vaisseau avec effets
    - Rendu du terrain et de la zone d'atterrissage
    - Affichage des messages (pause, résultat)

    SYSTÈME DE CACHE:
    Les images de fusée sont mises en cache par (puissance, angle, état)
    pour éviter de les recréer à chaque frame.

    ORDRE DE RENDU:
    1. effacer_tout(): Fond martien + étoiles
    2. dessiner_surface(): Terrain
    3. dessiner_vaisseau(): Vaisseau + particules
    4. ecrire_info(): HUD et informations

    Attributes:
        fenX, fenY (int): Dimensions de la fenêtre
        screen (Surface): Surface principale Pygame
        star_field (StarField): Système d'étoiles
        mars_bg (MarsBackground): Arrière-plan martien
        particle_system (ParticleSystem): Système de particules
        hud (SciFiHUD): Interface utilisateur
        image_cache (Dict): Cache des images de fusée
        rect (Rect): Rectangle de collision du vaisseau
        current_speed (int): Vitesse de simulation actuelle
    """

    def __init__(self):
        """
        Initialise Pygame et tous les sous-systèmes graphiques.
        """
        # === INITIALISATION PYGAME ===
        pygame.init()
        pygame.display.set_caption('Mars Lander IA - Realistic Edition')

        # Dimensions de la fenêtre (ajustées par l'échelle)
        self.fenX = fenX // echelle
        self.fenY = fenY // echelle
        self.screen = pygame.display.set_mode((self.fenX, self.fenY))

        # === SOUS-SYSTÈMES GRAPHIQUES ===
        self.star_field = StarField(self.fenX, self.fenY, 150)
        self.mars_bg = MarsBackground(self.fenX, self.fenY)
        self.particle_system = ParticleSystem()
        self.hud = SciFiHUD(self.fenX, self.fenY)

        # === CACHE D'IMAGES ===
        # Clé: (type, puissance, angle, détruit, posé)
        # Valeur: Surface Pygame
        self.image_cache: Dict[Tuple, pygame.Surface] = {}

        # === SURFACES PRÉ-RENDUES ===
        self.terrain_surface: Optional[pygame.Surface] = None
        self.zone_surface: Optional[pygame.Surface] = None

        # === ÉTAT DE LA ZONE D'ATTERRISSAGE ===
        self.landing_zone: Optional[Tuple] = None
        self.beacon_frame = 0  # Pour animation des balises

        # === ÉTAT GÉNÉRAL ===
        self.rect = None  # Rectangle de collision du vaisseau
        self.explosion_triggered = False  # Évite de rejouer l'explosion
        self.landing_triggered = False  # Évite de rejouer la poussière
        self.current_speed = 60  # Vitesse de simulation (FPS cible)

    def set_landing_zone(self, zone: Tuple) -> None:
        """
        Définit la zone d'atterrissage.

        Args:
            zone: Tuple ((x1, y), (x2, y)) des coordonnées de la zone
        """
        self.landing_zone = zone

    def create_rocket_image(self, power: int, destroyed: bool, landed: bool) -> pygame.Surface:
        """
        Crée une image de fusée dessinée procéduralement.

        La fusée est dessinée différemment selon son état:
        - En vol: fusée normale avec détails
        - Posée: fusée avec pieds d'atterrissage
        - Détruite: débris fumants

        Args:
            power: Niveau de puissance (0-4)
            destroyed: True si crashé
            landed: True si atterri

        Returns:
            Surface Pygame avec la fusée dessinée
        """
        # Dimensions de la surface
        width, height = 40, 60
        surface = pygame.Surface((width, height), pygame.SRCALPHA)

        if destroyed:
            # Dessin des débris
            self._draw_destroyed_rocket(surface, width, height)
        elif landed:
            # Dessin avec pieds d'atterrissage
            self._draw_rocket(surface, width, height, power, landed=True)
        else:
            # Dessin normal en vol
            self._draw_rocket(surface, width, height, power, landed=False)

        return surface

    def _draw_rocket(self, surface: pygame.Surface, w: int, h: int,
                     power: int, landed: bool = False) -> None:
        """
        Dessine une fusée stylisée procéduralement.

        STRUCTURE DE LA FUSÉE:
        - Cône rouge (nez)
        - Corps blanc/gris avec dégradé
        - Hublot bleu
        - Bande décorative rouge
        - Ailerons rouges
        - Tuyère grise
        - Pieds d'atterrissage (si posé)

        Args:
            surface: Surface cible
            w, h: Dimensions
            power: Niveau de puissance (non utilisé ici mais réservé)
            landed: True pour dessiner les pieds
        """
        cx = w // 2  # Centre X de la fusée

        # === PALETTE DE COULEURS ===
        BODY_WHITE = (240, 240, 250)    # Corps principal
        BODY_GRAY = (180, 180, 190)
        BODY_DARK = (100, 100, 110)     # Tuyère
        WINDOW_BLUE = (100, 180, 255)   # Hublot
        WINDOW_DARK = (50, 100, 150)
        FIN_RED = (200, 50, 50)         # Ailerons
        FIN_DARK = (150, 30, 30)
        NOSE_RED = (220, 60, 60)        # Nez

        # === CORPS PRINCIPAL (rectangle avec dégradé) ===
        body_top = 15
        body_bottom = 50
        body_width = 16

        # Dégradé horizontal pour donner du volume
        for i in range(body_width):
            # Plus sombre sur les bords, plus clair au centre
            t = abs(i - body_width // 2) / (body_width // 2)
            color = (
                int(BODY_WHITE[0] * (1 - t * 0.3)),
                int(BODY_WHITE[1] * (1 - t * 0.3)),
                int(BODY_WHITE[2] * (1 - t * 0.3))
            )
            pygame.draw.line(surface, color,
                           (cx - body_width // 2 + i, body_top),
                           (cx - body_width // 2 + i, body_bottom))

        # === CÔNE (NEZ) ===
        nose_points = [
            (cx, 2),          # Pointe
            (cx - 8, body_top),  # Base gauche
            (cx + 8, body_top)   # Base droite
        ]
        pygame.draw.polygon(surface, NOSE_RED, nose_points)
        # Reflet lumineux sur le cône
        pygame.draw.line(surface, (255, 150, 150), (cx - 2, 5), (cx - 4, body_top - 2), 2)

        # === HUBLOT ===
        window_y = 22
        pygame.draw.circle(surface, WINDOW_DARK, (cx, window_y), 5)  # Ombre
        pygame.draw.circle(surface, WINDOW_BLUE, (cx, window_y), 4)  # Vitre
        pygame.draw.circle(surface, (200, 230, 255), (cx - 1, window_y - 1), 2)  # Reflet

        # === BANDE DÉCORATIVE ===
        pygame.draw.rect(surface, FIN_RED, (cx - 8, 32, 16, 4))

        # === AILERONS (FINS) ===
        # Aileron gauche
        left_fin = [
            (cx - 8, 42),
            (cx - 18, 55),
            (cx - 8, 50)
        ]
        pygame.draw.polygon(surface, FIN_RED, left_fin)
        pygame.draw.polygon(surface, FIN_DARK, left_fin, 1)  # Contour

        # Aileron droit
        right_fin = [
            (cx + 8, 42),
            (cx + 18, 55),
            (cx + 8, 50)
        ]
        pygame.draw.polygon(surface, FIN_RED, right_fin)
        pygame.draw.polygon(surface, FIN_DARK, right_fin, 1)

        # === TUYÈRE (base du moteur) ===
        nozzle_points = [
            (cx - 6, body_bottom),
            (cx - 8, h - 5),
            (cx + 8, h - 5),
            (cx + 6, body_bottom)
        ]
        pygame.draw.polygon(surface, BODY_DARK, nozzle_points)
        pygame.draw.polygon(surface, (60, 60, 70), nozzle_points, 1)

        # === PIEDS D'ATTERRISSAGE (si posé) ===
        if landed:
            leg_color = (80, 80, 90)
            # Pied gauche
            pygame.draw.line(surface, leg_color, (cx - 6, h - 8), (cx - 15, h - 2), 2)
            pygame.draw.circle(surface, leg_color, (cx - 15, h - 2), 3)
            # Pied droit
            pygame.draw.line(surface, leg_color, (cx + 6, h - 8), (cx + 15, h - 2), 2)
            pygame.draw.circle(surface, leg_color, (cx + 15, h - 2), 3)

    def _draw_destroyed_rocket(self, surface: pygame.Surface, w: int, h: int) -> None:
        """
        Dessine les débris d'une fusée crashée.

        Affiche des morceaux éparpillés avec des points de braise
        pour simuler les restes fumants du crash.

        Args:
            surface: Surface cible
            w, h: Dimensions
        """
        cx = w // 2

        # === COULEURS DE DÉBRIS ===
        BURNT = (60, 50, 40)      # Métal brûlé
        DARK = (40, 35, 30)       # Contours
        ORANGE = (180, 80, 30)    # Braises
        EMBER = (255, 120, 40)    # Braises incandescentes

        # === MORCEAUX DE DÉBRIS ===
        # Polygones irréguliers simulant des fragments
        debris_points = [
            [(cx - 5, 15), (cx - 12, 35), (cx - 3, 40)],
            [(cx + 2, 10), (cx + 10, 30), (cx + 5, 38)],
            [(cx - 8, 38), (cx - 15, 50), (cx - 5, 52)],
            [(cx + 5, 40), (cx + 12, 55), (cx + 3, 50)]
        ]

        for debris in debris_points:
            pygame.draw.polygon(surface, BURNT, debris)
            pygame.draw.polygon(surface, DARK, debris, 1)

        # === POINTS DE BRAISE ===
        # Simule des points incandescents dans les débris
        ember_positions = [
            (cx - 8, 25, 2), (cx + 5, 30, 1), (cx - 3, 45, 2),
            (cx + 8, 48, 1), (cx, 35, 2)
        ]
        for bx, by, size in ember_positions:
            pygame.draw.circle(surface, ORANGE, (bx, by), size)
            pygame.draw.circle(surface, EMBER, (bx, by), max(1, size - 1))

    def get_rocket_image(self, power: int, angle: float,
                         destroyed: bool, landed: bool) -> pygame.Surface:
        """
        Récupère une image de fusée depuis le cache ou la crée.

        Utilise un cache pour éviter de recréer l'image à chaque frame.
        La clé du cache inclut tous les paramètres qui affectent l'apparence.

        Args:
            power: Niveau de puissance
            angle: Angle de rotation en degrés
            destroyed: État de destruction
            landed: État d'atterrissage

        Returns:
            Surface Pygame de la fusée (potentiellement rotée)
        """
        # Arrondir l'angle pour limiter la taille du cache
        angle_key = round(angle)
        cache_key = ('rocket', power, angle_key, destroyed, landed)

        if cache_key not in self.image_cache:
            # Créer l'image de base
            base_image = self.create_rocket_image(power, destroyed, landed)

            # Appliquer la rotation si nécessaire
            if angle != 0:
                rotated = pygame.transform.rotate(base_image, angle)
            else:
                rotated = base_image

            self.image_cache[cache_key] = rotated

        return self.image_cache[cache_key]

    def get_cached_image(self, path: str, angle: float) -> pygame.Surface:
        """
        Charge et cache une image depuis un fichier (méthode legacy).

        Conservée pour compatibilité avec d'éventuelles images externes.
        Utilise la fusée procédurale comme fallback si le fichier n'existe pas.

        Args:
            path: Chemin du fichier image
            angle: Angle de rotation

        Returns:
            Surface Pygame (chargée ou fallback)
        """
        angle_key = round(angle)
        cache_key = (path, angle_key)

        if cache_key not in self.image_cache:
            try:
                image = pygame.image.load(path).convert_alpha()
                if angle != 0:
                    image = pygame.transform.rotate(image, angle)
                self.image_cache[cache_key] = image
            except pygame.error:
                # Fallback: utiliser la fusée procédurale
                image = self.create_rocket_image(0, False, False)
                if angle != 0:
                    image = pygame.transform.rotate(image, angle)
                self.image_cache[cache_key] = image

        return self.image_cache[cache_key]

    def init_terrain(self, surface_mars: List[Tuple[int, int]], zone: Tuple) -> None:
        """
        Initialise le terrain avec les effets visuels.

        Pré-rend le terrain sur une surface pour éviter de le
        redessiner à chaque frame. Inclut:
        - Ligne principale du terrain
        - Zone d'atterrissage en vert
        - Ombres sous le terrain

        Args:
            surface_mars: Liste des points (x, y) du terrain
            zone: Tuple ((x1, y), (x2, y)) de la zone d'atterrissage
        """
        self.terrain_surface = pygame.Surface((self.fenX, self.fenY), pygame.SRCALPHA)
        self.landing_zone = zone

        # Dessiner chaque segment du terrain
        for i in range(len(surface_mars) - 1):
            # Coordonnées du segment (converties à l'échelle écran)
            x1, y1 = surface_mars[i]
            x1, y1 = x1 // echelle, y1 // echelle
            x2, y2 = surface_mars[i + 1]
            x2, y2 = x2 // echelle, y2 // echelle

            # Vérifier si c'est la zone d'atterrissage
            is_landing = (zone and
                         x1 >= zone[0][0] // echelle and
                         x2 <= zone[1][0] // echelle and
                         y1 == y2)  # Segment horizontal

            if is_landing:
                # Zone d'atterrissage: ligne verte épaisse
                color = ZONE_ATTERRISSAGE
                pygame.draw.line(self.terrain_surface, color, (x1, y1), (x2, y2), 4)
            else:
                # Terrain normal: dégradé avec ombre
                pygame.draw.line(self.terrain_surface, MARS_SOL_CLAIR, (x1, y1), (x2, y2), 8)
                pygame.draw.line(self.terrain_surface, MARS_SOL, (x1, y1 + 2), (x2, y2 + 2), 4)

            # Ombre sous le terrain
            pygame.draw.line(self.terrain_surface, MARS_OMBRE,
                           (x1, y1 + 8), (x2, y2 + 8), 3)

    def dessiner_surface(self, surface_mars: List[Tuple[int, int]]) -> None:
        """
        Dessine la surface de Mars avec les balises clignotantes.

        Utilise la surface pré-rendue et ajoute l'animation
        des balises lumineuses sur la zone d'atterrissage.

        Args:
            surface_mars: Liste des points du terrain (non utilisé si pré-rendu)
        """
        if self.terrain_surface is None:
            return
        self.screen.blit(self.terrain_surface, (0, 0))

        # === BALISES CLIGNOTANTES ===
        # Animation qui alterne toutes les 15 frames
        if self.landing_zone:
            self.beacon_frame += 1
            if (self.beacon_frame // 15) % 2 == 0:
                x1 = self.landing_zone[0][0] // echelle
                x2 = self.landing_zone[1][0] // echelle
                y = self.landing_zone[0][1] // echelle

                # Dessiner une balise à chaque extrémité de la zone
                for bx in [x1, x2]:
                    # Point lumineux central
                    pygame.draw.circle(self.screen, BALISE_LUMIERE, (bx, y - 5), 4)
                    # Halo autour
                    pygame.draw.circle(self.screen, ZONE_ATTERRISSAGE, (bx, y - 5), 6, 1)

    def dessiner_vaisseau(self, v, j) -> None:
        """
        Dessine le vaisseau avec tous ses effets visuels.

        ÉLÉMENTS DESSINÉS:
        1. Image du vaisseau (normale, posée ou détruite)
        2. Particules de propulsion (si moteur allumé)
        3. Lueur sous le vaisseau (si propulsion)
        4. Explosion (au moment du crash)
        5. Poussière (au moment de l'atterrissage)

        Args:
            v: Le vaisseau (objet Vaisseau)
            j: Le jeu (objet Jeu) - non utilisé mais requis par l'interface
        """
        # Position à l'écran (conversion des coordonnées monde -> écran)
        screen_x = v.x // echelle
        screen_y = v.y // echelle

        # === DESSIN DU VAISSEAU ===
        image = self.get_rocket_image(v.puissance, v.angle, v.detruit, v.est_pose)
        self.rect = image.get_rect(center=(screen_x, screen_y))
        self.screen.blit(image, self.rect)

        # === EFFETS DE PROPULSION ===
        if not v.detruit and not v.est_pose and v.puissance > 0:
            # Position de la flamme (sous le vaisseau, dans la direction de la poussée)
            flame_x = screen_x + math.sin(math.radians(v.angle)) * 20
            flame_y = screen_y + math.cos(math.radians(v.angle)) * 20

            # Émettre des particules de flamme
            self.particle_system.emit_thrust(flame_x, flame_y, v.angle, v.puissance)

            # Lueur diffuse sous le vaisseau
            glow_size = v.puissance * 8
            glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*FLAMME_MILIEU, 50),
                             (glow_size, glow_size), glow_size)
            # Mode BLEND_ADD pour effet lumineux
            self.screen.blit(glow_surf,
                           (flame_x - glow_size, flame_y - glow_size),
                           special_flags=pygame.BLEND_ADD)

        # === EFFET D'EXPLOSION (unique) ===
        if v.detruit and not self.explosion_triggered:
            self.particle_system.emit_explosion(screen_x, screen_y, 50)
            self.explosion_triggered = True  # Ne pas rejouer

        # === EFFET D'ATTERRISSAGE (unique) ===
        if v.est_pose and not self.landing_triggered:
            self.particle_system.emit_dust(screen_x, screen_y + 15, 30)
            self.landing_triggered = True  # Ne pas rejouer

        # === RESET DES TRIGGERS ===
        # Quand le vaisseau est relancé, permettre de nouveaux effets
        if not v.detruit and not v.est_pose:
            self.explosion_triggered = False
            self.landing_triggered = False

    def effacer_tout(self) -> None:
        """
        Efface l'écran et dessine le fond (première couche).

        Dessine dans l'ordre:
        1. Arrière-plan martien (dégradé + montagnes)
        2. Champ d'étoiles avec mise à jour de l'animation
        """
        # Arrière-plan martien
        self.mars_bg.draw(self.screen)

        # Étoiles (avec animation de scintillement)
        self.star_field.update()
        self.star_field.draw(self.screen)

    def ecrire_info(self, v, ia, j) -> None:
        """
        Affiche le HUD (Head-Up Display) avec toutes les informations.

        PANNEAUX AFFICHÉS:
        1. STATUT (gauche): Position, vitesses, angle, puissance, carburant
        2. IA LEARNING (droite): Récompense, Q-Table, epsilon, épisodes
        3. VITESSE (bas gauche): Vitesse de simulation actuelle
        4. MINI-RADAR (bas droite): Position relative à la zone

        ALERTES:
        - "CARBURANT FAIBLE" si fuel < 100
        - "VITESSE CRITIQUE" si va_trop_vite()

        MESSAGES SPÉCIAUX:
        - Message de pause
        - Résultat (succès/crash)

        Args:
            v: Le vaisseau
            ia: L'agent IA
            j: Le jeu
        """
        self.hud._init_fonts()
        self.hud.update()  # Animation du HUD

        # =================================================================
        # PANNEAU STATUT (gauche)
        # =================================================================
        self.hud.draw_panel(self.screen, 10, 10, 200, 180, "STATUT")

        y_offset = 40
        # Position
        self.hud.draw_value(self.screen, 20, y_offset, "Position X", int(v.x))
        self.hud.draw_value(self.screen, 20, y_offset + 18, "Position Y", int(v.y))

        # Vitesses (en rouge si trop rapide)
        self.hud.draw_value(self.screen, 20, y_offset + 36, "Vitesse H", round(v.h_speed, 1),
                          HUD_DANGER if v.va_trop_vite_h() else HUD_TEXTE)
        self.hud.draw_value(self.screen, 20, y_offset + 54, "Vitesse V", round(v.v_speed, 1),
                          HUD_DANGER if v.va_trop_vite_v() else HUD_TEXTE)

        # Contrôles
        self.hud.draw_value(self.screen, 20, y_offset + 72, "Angle", int(v.angle))
        self.hud.draw_value(self.screen, 20, y_offset + 90, "Puissance", v.puissance)

        # Barre de carburant (couleur selon le niveau)
        fuel_color = HUD_PRIMAIRE if v.fuel > 200 else (HUD_ALERTE if v.fuel > 100 else HUD_DANGER)
        self.hud.draw_progress_bar(self.screen, 20, y_offset + 115, 170, 15,
                                  v.fuel, 1000, fuel_color)
        fuel_text = self.hud.font_small.render(f"Fuel: {int(v.fuel)}", True, HUD_TEXTE)
        self.screen.blit(fuel_text, (20, y_offset + 132))

        # =================================================================
        # INDICATEUR DE VITESSE DE SIMULATION (bas gauche)
        # =================================================================
        speed_panel_y = self.fenY - 50
        self.hud.draw_panel(self.screen, 10, speed_panel_y, 150, 40, "")
        speed_text = self.hud.font_medium.render(f"Vitesse: x{self.current_speed}", True, HUD_ACCENT)
        self.screen.blit(speed_text, (20, speed_panel_y + 10))
        hint_text = self.hud.font_small.render("+/- pour ajuster", True, HUD_SECONDAIRE)
        self.screen.blit(hint_text, (20, speed_panel_y + 25))

        # =================================================================
        # PANNEAU IA (droite) - Seulement si IA active
        # =================================================================
        if ia_active:
            self.hud.draw_panel(self.screen, self.fenX - 220, 10, 210, 160, "IA LEARNING")

            y_offset = 40
            self.hud.draw_value(self.screen, self.fenX - 210, y_offset, "Reward", round(ia.recompense, 1))
            self.hud.draw_value(self.screen, self.fenX - 210, y_offset + 18, "Cumul", round(ia.recup_recompense(), 1))
            self.hud.draw_value(self.screen, self.fenX - 210, y_offset + 36, "Q-Table", len(ia.q_table))
            self.hud.draw_value(self.screen, self.fenX - 210, y_offset + 54, "Epsilon", round(ia.epsilon, 4))
            self.hud.draw_value(self.screen, self.fenX - 210, y_offset + 72, "Episodes", j.tentative)
            self.hud.draw_value(self.screen, self.fenX - 210, y_offset + 90, "Succès", j.att_reussi,
                              SUCCES_VERT if j.att_reussi > 0 else HUD_TEXTE)

        # =================================================================
        # MINI-RADAR (bas droite)
        # =================================================================
        if self.landing_zone:
            zone_center = ((self.landing_zone[0][0] + self.landing_zone[1][0]) / 2,
                          self.landing_zone[0][1])
            zone_width = self.landing_zone[1][0] - self.landing_zone[0][0]
            self.hud.draw_mini_radar(self.screen, self.fenX - 110, self.fenY - 120, 100,
                                    (v.x, v.y), zone_center, zone_width)

        # =================================================================
        # ALERTES
        # =================================================================
        if v.fuel < 100 and not v.detruit and not v.est_pose:
            self.hud.draw_alert(self.screen, self.fenX // 2 - 80, 50, "CARBURANT FAIBLE", "danger")

        if v.va_trop_vite() and not v.detruit and not v.est_pose:
            self.hud.draw_alert(self.screen, self.fenX // 2 - 80, 75, "VITESSE CRITIQUE", "warning")

        # =================================================================
        # MESSAGES SPÉCIAUX
        # =================================================================
        if j.paused:
            self._afficher_message_pause()

        if v.est_pose or v.detruit:
            self._afficher_resultat(v)

        # =================================================================
        # PARTICULES (dernière couche, par-dessus tout)
        # =================================================================
        self.particle_system.update()
        self.particle_system.draw(self.screen)

    def _afficher_message_pause(self) -> None:
        """
        Affiche l'écran de pause.

        Ajoute un overlay sombre semi-transparent et
        le texte "PAUSE" au centre de l'écran.
        """
        # Overlay sombre
        overlay = pygame.Surface((self.fenX, self.fenY), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))  # Noir avec alpha 150
        self.screen.blit(overlay, (0, 0))

        # Texte "PAUSE"
        font = pygame.font.Font(None, 72)
        text = font.render("PAUSE", True, HUD_ACCENT)
        text_rect = text.get_rect(center=(self.fenX // 2, self.fenY // 2))
        self.screen.blit(text, text_rect)

        # Instruction pour reprendre
        font_small = pygame.font.Font(None, 28)
        hint = font_small.render("Appuyez sur P pour continuer", True, HUD_TEXTE)
        hint_rect = hint.get_rect(center=(self.fenX // 2, self.fenY // 2 + 50))
        self.screen.blit(hint, hint_rect)

    def _afficher_resultat(self, v) -> None:
        """
        Affiche le résultat de la tentative (succès ou crash).

        Affiche un message différent selon le résultat:
        - Succès: "ATTERRISSAGE RÉUSSI" en vert avec lueur
        - Crash: "CRASH" en rouge avec lueur

        Args:
            v: Le vaisseau (pour connaître l'état)
        """
        font = pygame.font.Font(None, 56)
        font_small = pygame.font.Font(None, 28)

        if v.est_pose:
            # === SUCCÈS ===
            text = font.render("ATTERRISSAGE RÉUSSI", True, SUCCES_VERT)
            # Effet de lueur verte
            glow = pygame.Surface((400, 80), pygame.SRCALPHA)
            pygame.draw.rect(glow, (*SUCCES_VERT, 30), (0, 0, 400, 80))
            self.screen.blit(glow, (self.fenX // 2 - 200, self.fenY // 2 - 70))
        else:
            # === CRASH ===
            text = font.render("CRASH", True, CRASH_ROUGE)
            # Effet de lueur rouge
            glow = pygame.Surface((200, 80), pygame.SRCALPHA)
            pygame.draw.rect(glow, (*CRASH_ROUGE, 30), (0, 0, 200, 80))
            self.screen.blit(glow, (self.fenX // 2 - 100, self.fenY // 2 - 70))

        # Afficher le texte centré
        text_rect = text.get_rect(center=(self.fenX // 2, self.fenY // 2 - 40))
        self.screen.blit(text, text_rect)

        # Instruction pour recommencer
        hint = font_small.render("ESPACE pour recommencer", True, HUD_TEXTE)
        hint_rect = hint.get_rect(center=(self.fenX // 2, self.fenY // 2 + 20))
        self.screen.blit(hint, hint_rect)

    def affiche_info(self, nom: str, valeur: any, pos: Tuple[int, int]) -> None:
        """
        Méthode legacy pour compatibilité.

        Conservée pour ne pas casser le code existant mais
        remplacée par le système HUD moderne.
        """
        pass

    # ==========================================================================
    # TRAJECTOIRE PRÉDICTIVE
    # ==========================================================================

    def dessiner_trajectoire(self, v, vent_force: Tuple[float, float] = (0, 0)) -> None:
        """
        Dessine une trajectoire prédictive montrant où le vaisseau va atterrir.

        Simule la physique du vaisseau sur plusieurs frames pour prédire
        sa trajectoire future. La couleur change si le vaisseau va trop vite.

        Args:
            v: Le vaisseau (pour position, vitesse, angle, puissance actuels)
            vent_force: Force du vent actuelle (vx, vy)
        """
        if not trajectoire_active or v.detruit or v.est_pose:
            return

        # Copier l'état actuel du vaisseau pour la simulation
        sim_x = v.x
        sim_y = v.y
        sim_vx = v.h_speed
        sim_vy = v.v_speed

        # Points de la trajectoire
        points = []
        points.append((int(sim_x // echelle), int(sim_y // echelle)))

        # Simuler la trajectoire
        for _ in range(trajectoire_points):
            # Appliquer la gravité
            sim_vy += gravite

            # Appliquer le vent
            sim_vx += vent_force[0]
            sim_vy += vent_force[1]

            # Mettre à jour la position
            sim_x += sim_vx
            sim_y += sim_vy

            # Convertir en coordonnées écran
            screen_x = int(sim_x // echelle)
            screen_y = int(sim_y // echelle)

            # Arrêter si hors écran
            if screen_y > self.fenY or screen_x < 0 or screen_x > self.fenX:
                break

            points.append((screen_x, screen_y))

        # Dessiner la trajectoire si assez de points
        if len(points) > 1:
            # Couleur selon la vitesse (danger si trop rapide)
            speed = math.sqrt(v.h_speed**2 + v.v_speed**2)
            if speed > max_v_speed or abs(v.h_speed) > max_h_speed:
                color = TRAJECTOIRE_DANGER[:3]
            else:
                color = TRAJECTOIRE[:3]

            # Dessiner des points espacés pour effet pointillé
            for i, point in enumerate(points):
                if i % 3 == 0:  # Un point sur 3 pour effet pointillé
                    alpha = int(200 * (1 - i / len(points)))  # Fondu progressif
                    pygame.draw.circle(self.screen, color, point, 2)

    # ==========================================================================
    # GRAPHIQUE D'APPRENTISSAGE
    # ==========================================================================

    def dessiner_graphique_apprentissage(self, ia) -> None:
        """
        Dessine un graphique montrant l'évolution des récompenses.

        Affiche les récompenses des derniers épisodes sous forme
        de graphique linéaire dans le coin inférieur droit.

        Args:
            ia: L'agent IA (pour récupérer l'historique des récompenses)
        """
        if not graphique_actif or not ia_active:
            return

        # Récupérer les récompenses récentes
        rewards = ia.get_recent_rewards(graphique_points)
        if len(rewards) < 2:
            return

        # Dimensions et position du graphique
        graph_width = 180
        graph_height = 80
        graph_x = self.fenX - graph_width - 10
        graph_y = self.fenY - graph_height - 130  # Au-dessus du radar

        # Fond du graphique
        graph_surface = pygame.Surface((graph_width, graph_height), pygame.SRCALPHA)
        graph_surface.fill((*HUD_FOND, 180))
        self.screen.blit(graph_surface, (graph_x, graph_y))

        # Bordure
        pygame.draw.rect(self.screen, HUD_BORDURE,
                        (graph_x, graph_y, graph_width, graph_height), 1)

        # Titre
        title = self.hud.font_small.render("Récompenses", True, HUD_ACCENT)
        self.screen.blit(title, (graph_x + 5, graph_y + 2))

        # Calculer l'échelle du graphique
        min_reward = min(rewards)
        max_reward = max(rewards)
        reward_range = max_reward - min_reward if max_reward != min_reward else 1

        # Zone de dessin du graphique (avec marge)
        plot_x = graph_x + 5
        plot_y = graph_y + 18
        plot_width = graph_width - 10
        plot_height = graph_height - 25

        # Ligne de base (zéro)
        if min_reward < 0 < max_reward:
            zero_y = plot_y + plot_height - int((-min_reward / reward_range) * plot_height)
            pygame.draw.line(self.screen, HUD_SECONDAIRE,
                           (plot_x, zero_y), (plot_x + plot_width, zero_y), 1)

        # Dessiner la courbe des récompenses
        points = []
        for i, reward in enumerate(rewards):
            x = plot_x + int(i / len(rewards) * plot_width)
            y = plot_y + plot_height - int((reward - min_reward) / reward_range * plot_height)
            points.append((x, y))

        if len(points) > 1:
            # Couleur selon la tendance (vert si amélioration, rouge sinon)
            avg_recent = sum(rewards[-10:]) / min(10, len(rewards))
            avg_old = sum(rewards[:10]) / min(10, len(rewards)) if len(rewards) > 10 else avg_recent

            if avg_recent > avg_old:
                line_color = SUCCES_VERT
            else:
                line_color = HUD_ALERTE

            pygame.draw.lines(self.screen, line_color, False, points, 2)

        # Afficher la moyenne
        avg = sum(rewards) / len(rewards)
        avg_text = self.hud.font_small.render(f"Moy: {avg:.1f}", True, HUD_TEXTE)
        self.screen.blit(avg_text, (graph_x + 5, graph_y + graph_height - 15))

    # ==========================================================================
    # INDICATEUR DE VENT
    # ==========================================================================

    def dessiner_indicateur_vent(self, vent_force: Tuple[float, float]) -> None:
        """
        Dessine un indicateur montrant la direction et force du vent.

        Args:
            vent_force: Force du vent (vx, vy)
        """
        if not vent_actif:
            return

        # Position de l'indicateur (en haut au centre)
        center_x = self.fenX // 2
        center_y = 30

        # Calculer la magnitude et l'angle du vent
        vx, vy = vent_force
        magnitude = math.sqrt(vx**2 + vy**2)

        if magnitude < 0.01:
            # Pas de vent significatif
            return

        # Normaliser et mise à l'échelle pour l'affichage
        scale = 30  # Longueur max de la flèche
        arrow_length = min(magnitude * 100, scale)
        angle = math.atan2(vy, vx)

        # Point de fin de la flèche
        end_x = center_x + int(math.cos(angle) * arrow_length)
        end_y = center_y + int(math.sin(angle) * arrow_length)

        # Dessiner le fond circulaire
        pygame.draw.circle(self.screen, (*HUD_FOND, 150), (center_x, center_y), 25)
        pygame.draw.circle(self.screen, HUD_BORDURE, (center_x, center_y), 25, 1)

        # Dessiner la flèche du vent
        wind_color = HUD_ACCENT if magnitude < 0.2 else HUD_ALERTE
        pygame.draw.line(self.screen, wind_color, (center_x, center_y), (end_x, end_y), 2)

        # Pointe de la flèche
        arrow_angle = 0.5
        arrow_size = 8
        pygame.draw.line(self.screen, wind_color, (end_x, end_y),
                        (end_x - int(math.cos(angle - arrow_angle) * arrow_size),
                         end_y - int(math.sin(angle - arrow_angle) * arrow_size)), 2)
        pygame.draw.line(self.screen, wind_color, (end_x, end_y),
                        (end_x - int(math.cos(angle + arrow_angle) * arrow_size),
                         end_y - int(math.sin(angle + arrow_angle) * arrow_size)), 2)

        # Label "VENT"
        label = self.hud.font_small.render("VENT", True, HUD_SECONDAIRE)
        self.screen.blit(label, (center_x - 15, center_y + 28))
