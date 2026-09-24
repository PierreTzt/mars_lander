"""
Module de gestion du vaisseau spatial pour Mars Lander.

Ce module implémente la simulation physique du vaisseau spatial,
incluant la gravité, la propulsion, et les conditions d'atterrissage.

PHYSIQUE SIMULÉE:
=================

1. GRAVITÉ:
   - Le vaisseau subit une accélération constante vers le bas
   - v_speed += gravite à chaque frame

2. PROPULSION:
   - La poussée s'applique dans la direction opposée au nez du vaisseau
   - La force est décomposée en composantes verticale et horizontale
   - v_speed -= puissance * cos(angle)
   - h_speed -= puissance * sin(angle)

3. CARBURANT:
   - Chaque niveau de puissance consomme du carburant
   - fuel -= puissance à chaque frame
   - Sans carburant, pas de poussée possible

SYSTÈME DE COORDONNÉES:
======================
- X: position horizontale (0 = gauche, fenX = droite)
- Y: position verticale (0 = haut, fenY = bas)
- Angle: rotation (-90° = gauche, 0° = vertical, +90° = droite)
- h_speed: négatif = gauche, positif = droite
- v_speed: négatif = monte, positif = descend
"""

import math
from typing import Dict, Any

# Import des constantes de configuration
from .data import (
    fenX, fenY,           # Dimensions du monde
    gravite,              # Force de gravité
    max_h_speed,          # Vitesse horizontale max pour atterrir
    max_v_speed,          # Vitesse verticale max pour atterrir
    angle_vaisseau_max    # Angle de rotation maximum (±90°)
)


class Vaisseau:
    """
    Représente le vaisseau spatial avec sa physique complète.

    Le vaisseau est contrôlé par deux paramètres:
    - angle: orientation du vaisseau (-90° à +90°)
    - puissance: force des moteurs (0 à 4)

    L'objectif est d'atterrir sur la zone plate avec:
    - angle = 0° (horizontal)
    - vitesse verticale ≤ 40 m/s
    - vitesse horizontale ≤ 20 m/s

    Attributes:
        x (float): Position horizontale dans le monde
        y (float): Position verticale (0 = haut)
        h_speed (float): Vitesse horizontale (pixels/frame)
        v_speed (float): Vitesse verticale (pixels/frame)
        fuel (float): Carburant restant
        angle (float): Angle d'inclinaison (-90 à +90 degrés)
        puissance (int): Niveau de puissance des moteurs (0-4)
        detruit (bool): True si le vaisseau a crashé
        est_pose (bool): True si le vaisseau est posé avec succès
    """

    def __init__(self):
        """
        Initialise le vaisseau avec des valeurs par défaut.

        Ces valeurs seront écrasées par init_vaisseau() avec
        les paramètres du scénario choisi.
        """
        # Position dans le monde
        self.x: float = 0       # Position horizontale
        self.y: float = 0       # Position verticale

        # Vitesse (pixels par frame)
        self.h_speed: float = 0 # Vitesse horizontale
        self.v_speed: float = 0 # Vitesse verticale

        # Ressources et contrôles
        self.fuel: float = 0    # Carburant restant
        self.angle: float = 0   # Angle d'inclinaison (-90° à +90°)
        self.puissance: int = 0 # Puissance des moteurs (0-4)

        # État du vaisseau
        self.detruit: bool = False  # Crashé?
        self.est_pose: bool = False # Atterri avec succès?

    def init_vaisseau(self, info: Dict[str, Any]) -> None:
        """
        Initialise le vaisseau avec les paramètres du scénario.

        Cette méthode est appelée au début du jeu et à chaque
        relancement pour remettre le vaisseau à sa position initiale.

        Args:
            info: Dictionnaire contenant:
                - 'x': Position horizontale initiale
                - 'y': Position verticale initiale
                - 'h_speed': Vitesse horizontale initiale
                - 'v_speed': Vitesse verticale initiale
                - 'fuel': Carburant initial
                - 'rotate': Angle initial
                - 'power': Puissance initiale
        """
        self.x = info['x']
        self.y = info['y']
        self.h_speed = info['h_speed']
        self.v_speed = info['v_speed']
        self.fuel = info['fuel']
        self.angle = info['rotate']
        self.puissance = info['power']

        # Réinitialiser l'état
        self.detruit = False
        self.est_pose = False

    # ==========================================================================
    # VÉRIFICATIONS DE LIMITES
    # ==========================================================================

    def en_dehors_de_la_zone(self) -> bool:
        """
        Vérifie si le vaisseau est sorti de la zone de jeu.

        Si le vaisseau sort de l'écran (en haut, en bas, à gauche ou à droite),
        il sera considéré comme perdu.

        Returns:
            True si le vaisseau est hors des limites du monde
        """
        return not (0 < self.x < fenX and 0 < self.y < fenY)

    def verif_si_HS(self) -> bool:
        """
        Vérifie si le vaisseau est "hors service" (détruit).

        Appelé à chaque frame pour vérifier si le vaisseau
        est sorti des limites du monde.

        Returns:
            True si le vaisseau vient d'être détruit
        """
        if self.en_dehors_de_la_zone():
            self.detruit = True
            return True
        return False

    def a_plus_dessence(self) -> bool:
        """
        Vérifie si le vaisseau n'a plus de carburant.

        Sans carburant, les moteurs ne peuvent plus fonctionner
        et le vaisseau subit uniquement la gravité.

        Returns:
            True si le carburant est épuisé (≤ 0)
        """
        if self.fuel <= 0:
            self.fuel = 0  # Éviter les valeurs négatives
            return True
        return False

    # ==========================================================================
    # VÉRIFICATIONS D'ORIENTATION
    # ==========================================================================

    def est_droit(self) -> bool:
        """
        Vérifie si le vaisseau est approximativement horizontal.

        Utile pour donner des bonus de récompense quand le vaisseau
        approche de l'orientation correcte pour atterrir.

        Returns:
            True si l'angle est entre -30° et +30°
        """
        return -30 <= self.angle <= 30

    def est_parfaitement_droit(self) -> bool:
        """
        Vérifie si le vaisseau est parfaitement horizontal.

        C'est une condition OBLIGATOIRE pour atterrir avec succès.
        L'angle doit être exactement 0°.

        Returns:
            True si l'angle est exactement 0°
        """
        return self.angle == 0

    # ==========================================================================
    # SIMULATION PHYSIQUE
    # ==========================================================================

    def actualisation(self) -> None:
        """
        Met à jour la physique du vaisseau pour une frame.

        ÉTAPES DE CALCUL:
        1. Vérifier si la poussée est possible (carburant, pas détruit)
        2. Consommer le carburant
        3. Appliquer la gravité
        4. Limiter l'angle aux bornes autorisées
        5. Calculer et appliquer la poussée
        6. Mettre à jour la position

        FORMULES DE POUSSÉE:
        - La poussée est dans la direction opposée au nez
        - Composante verticale: -puissance * cos(angle)
        - Composante horizontale: -puissance * sin(angle)
        """
        # Pas de poussée si plus de carburant ou jeu terminé
        if self.fuel <= 0 or self.detruit or self.est_pose:
            self.puissance = 0

        # Ne rien faire si le jeu est terminé
        if self.detruit or self.est_pose:
            return

        # === CONSOMMATION DE CARBURANT ===
        # Chaque niveau de puissance consomme 1 unité de fuel par frame
        self.fuel -= self.puissance

        # === APPLICATION DE LA GRAVITÉ ===
        # La gravité ajoute une accélération vers le bas (Y positif)
        self.v_speed += gravite

        # === LIMITATION DE L'ANGLE ===
        # L'angle ne peut pas dépasser ±90° (ou la valeur configurée)
        if angle_vaisseau_max != 0:
            self.angle = max(-angle_vaisseau_max,
                           min(angle_vaisseau_max, self.angle))

        # === CALCUL DE LA POUSSÉE ===
        # Convertir l'angle en radians pour les fonctions trigonométriques
        angle_radians = math.radians(self.angle)

        # sin(angle) donne la composante horizontale
        # cos(angle) donne la composante verticale
        sin_angle = math.sin(angle_radians)
        cos_angle = math.cos(angle_radians)

        # === APPLICATION DE LA POUSSÉE ===
        # La poussée pousse dans la direction opposée au nez
        # - Pour un angle de 0° (vertical): toute la poussée va vers le haut
        # - Pour un angle positif: une partie de la poussée va vers la gauche
        # - Pour un angle négatif: une partie de la poussée va vers la droite
        self.v_speed -= self.puissance * cos_angle
        self.h_speed -= self.puissance * sin_angle

        # === MISE À JOUR DE LA POSITION ===
        # Intégration simple: position += vitesse
        self.y += self.v_speed
        self.x += self.h_speed

    # ==========================================================================
    # CONDITIONS D'ATTERRISSAGE
    # ==========================================================================

    def peut_atterir(self) -> bool:
        """
        Vérifie si toutes les conditions d'atterrissage sont remplies.

        CONDITIONS REQUISES:
        1. Angle = 0° (vaisseau parfaitement horizontal)
        2. Vitesse verticale ≤ 40 m/s
        3. Vitesse horizontale ≤ 20 m/s

        Si ces conditions sont remplies ET que le vaisseau touche
        la zone d'atterrissage, c'est un succès. Sinon, c'est un crash.

        Returns:
            True si le vaisseau peut atterrir en sécurité
        """
        return self.angle == 0 and not self.va_trop_vite()

    def va_trop_vite(self) -> bool:
        """
        Vérifie si le vaisseau va trop vite pour atterrir.

        Les limites sont définies dans data.py:
        - max_v_speed = 40 (vitesse verticale max)
        - max_h_speed = 20 (vitesse horizontale max)

        Returns:
            True si une des vitesses dépasse sa limite
        """
        return abs(self.v_speed) >= max_v_speed or abs(self.h_speed) >= max_h_speed

    def va_trop_vite_h(self) -> bool:
        """
        Vérifie si la vitesse horizontale est excessive.

        Utilisé pour le système de récompenses et l'affichage HUD.

        Returns:
            True si |h_speed| ≥ max_h_speed (20 m/s)
        """
        return abs(self.h_speed) >= max_h_speed

    def va_trop_vite_v(self) -> bool:
        """
        Vérifie si la vitesse verticale est excessive.

        Utilisé pour le système de récompenses et l'affichage HUD.

        Returns:
            True si |v_speed| ≥ max_v_speed (40 m/s)
        """
        return abs(self.v_speed) >= max_v_speed

    # ==========================================================================
    # UTILITAIRES
    # ==========================================================================

    def get_state(self) -> Dict[str, Any]:
        """
        Retourne l'état complet du vaisseau sous forme de dictionnaire.

        Utile pour le débogage, la sauvegarde, ou l'affichage.

        Returns:
            Dictionnaire avec toutes les propriétés du vaisseau:
            - x, y: Position
            - h_speed, v_speed: Vitesses
            - fuel: Carburant restant
            - angle: Orientation
            - puissance: Niveau de poussée
            - detruit: État de destruction
            - est_pose: État d'atterrissage
        """
        return {
            "x": self.x,
            "y": self.y,
            "h_speed": self.h_speed,
            "v_speed": self.v_speed,
            "fuel": self.fuel,
            "angle": self.angle,
            "puissance": self.puissance,
            "detruit": self.detruit,
            "est_pose": self.est_pose
        }
