"""
Module de gestion de la surface martienne pour Mars Lander.

Ce module gère le terrain de Mars et la zone d'atterrissage.
Il fournit des méthodes pour analyser la position du vaisseau
par rapport à la zone cible.

TERRAIN DE MARS:
================
Le terrain est défini comme une liste de points (x, y) qui forment
une ligne brisée. Le vaisseau crash s'il touche cette ligne
en dehors de la zone d'atterrissage ou avec une vitesse excessive.

ZONE D'ATTERRISSAGE:
====================
C'est le seul endroit où le vaisseau peut se poser.
Elle est automatiquement détectée comme le segment HORIZONTAL
du terrain (deux points consécutifs avec la même altitude Y).

CONDITIONS D'ATTERRISSAGE RÉUSSI:
- Le vaisseau doit toucher le terrain DANS la zone d'atterrissage
- L'angle doit être exactement 0° (horizontal)
- La vitesse verticale doit être ≤ 40 m/s
- La vitesse horizontale doit être ≤ 20 m/s
"""

from typing import List, Tuple, Optional


class Surface:
    """
    Représente la surface de Mars avec le terrain et la zone d'atterrissage.

    Cette classe analyse la position du vaisseau par rapport à la zone
    d'atterrissage pour guider le système de récompenses de l'IA.

    Attributes:
        mars_surface (List): Liste des points (x, y) définissant le terrain
        atterissage (Tuple): Les deux points délimitant la zone d'atterrissage
            Format: ((x1, y1), (x2, y2)) où y1 == y2 (segment horizontal)
    """

    def __init__(self, mars_surface: List[Tuple[int, int]]):
        """
        Initialise la surface avec les points du terrain.

        Args:
            mars_surface: Liste des coordonnées (x, y) du terrain
                          Les points sont ordonnés de gauche à droite
        """
        self.mars_surface = mars_surface
        self.atterissage: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None

    def calcul_zone_atterissage(self, scenar: dict) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Détecte et enregistre la zone d'atterrissage.

        La zone d'atterrissage est définie comme le premier segment
        HORIZONTAL du terrain (où deux points consécutifs ont le même Y).

        ALGORITHME:
        1. Parcourir les points du terrain par paires
        2. Trouver la première paire où y1 == y2
        3. Ces deux points définissent la zone d'atterrissage

        Args:
            scenar: Configuration du scénario contenant 'surface_mars'

        Returns:
            Tuple des deux points: ((x1, y), (x2, y))
        """
        surf = scenar['surface_mars']

        # Parcourir tous les segments du terrain
        for i in range(len(surf) - 1):
            p1_x, p1_y = surf[i]       # Point de départ du segment
            p2_x, p2_y = surf[i + 1]   # Point de fin du segment

            # Segment horizontal = zone d'atterrissage
            if p1_y == p2_y:
                self.atterissage = ((p1_x, p1_y), (p2_x, p2_y))
                break

        return self.atterissage

    # ==========================================================================
    # ANALYSE DE POSITION HORIZONTALE
    # ==========================================================================

    def est_dans_la_zone(self, v) -> bool:
        """
        Vérifie si le vaisseau est horizontalement au-dessus de la zone.

        Le vaisseau doit être entre les coordonnées X de la zone
        pour pouvoir atterrir avec succès.

        Args:
            v: Le vaisseau (objet Vaisseau)

        Returns:
            True si atterissage[0].x < v.x < atterissage[1].x
        """
        return self.atterissage[0][0] < v.x < self.atterissage[1][0]

    def est_a_gauche_de_la_zone(self, v) -> bool:
        """
        Vérifie si le vaisseau est à gauche de la zone d'atterrissage.

        Args:
            v: Le vaisseau

        Returns:
            True si le vaisseau est à gauche du bord gauche de la zone
        """
        return v.x < self.atterissage[0][0]

    def est_a_droite_de_la_zone(self, v) -> bool:
        """
        Vérifie si le vaisseau est à droite de la zone d'atterrissage.

        Args:
            v: Le vaisseau

        Returns:
            True si le vaisseau est à droite du bord droit de la zone
        """
        return v.x > self.atterissage[1][0]

    # ==========================================================================
    # ANALYSE DE POSITION VERTICALE
    # ==========================================================================

    def est_en_haut_de_la_zone(self, v) -> bool:
        """
        Vérifie si le vaisseau est au-dessus de l'altitude de la zone.

        Note: Dans notre système de coordonnées, Y=0 est en haut.
        Donc "au-dessus" = Y plus petit.

        Args:
            v: Le vaisseau

        Returns:
            True si le vaisseau est plus haut (Y plus petit) que la zone
        """
        return v.y < self.atterissage[0][1]

    def est_en_bas_de_la_zone(self, v) -> bool:
        """
        Vérifie si le vaisseau est en-dessous de l'altitude de la zone.

        Cela signifie que le vaisseau a dépassé la zone sans atterrir
        (ce qui est normalement impossible car il aurait crashé).

        Args:
            v: Le vaisseau

        Returns:
            True si le vaisseau est plus bas (Y plus grand) que la zone
        """
        return v.y > self.atterissage[1][1]

    # ==========================================================================
    # ANALYSE DE DIRECTION
    # ==========================================================================

    def va_a_gauche(self, v) -> bool:
        """
        Vérifie si le vaisseau se déplace vers la gauche.

        Args:
            v: Le vaisseau

        Returns:
            True si h_speed < 0 (vitesse horizontale négative)
        """
        return v.h_speed < 0

    def va_a_droite(self, v) -> bool:
        """
        Vérifie si le vaisseau se déplace vers la droite.

        Args:
            v: Le vaisseau

        Returns:
            True si h_speed > 0 (vitesse horizontale positive)
        """
        return v.h_speed > 0

    def va_en_haut(self, v) -> bool:
        """
        Vérifie si le vaisseau monte.

        Note: Dans notre système, Y augmente vers le bas,
        donc "monter" = vitesse verticale négative.

        Args:
            v: Le vaisseau

        Returns:
            True si v_speed < 0 (le vaisseau monte)
        """
        return v.v_speed < 0

    def va_en_bas(self, v) -> bool:
        """
        Vérifie si le vaisseau descend.

        C'est le comportement normal sous l'effet de la gravité.

        Args:
            v: Le vaisseau

        Returns:
            True si v_speed > 0 (le vaisseau descend)
        """
        return v.v_speed > 0

    # ==========================================================================
    # ANALYSE D'APPROCHE
    # ==========================================================================

    def se_rapproche_de_la_zone(self, v) -> bool:
        """
        Vérifie si le vaisseau se rapproche de la zone d'atterrissage.

        Cette méthode est CRUCIALE pour le système de récompenses.
        Elle analyse la position ET la direction du vaisseau pour
        déterminer s'il progresse vers l'objectif.

        LOGIQUE PAR POSITION:
        =====================

        1. DANS LA ZONE (horizontalement):
           → Se rapproche s'il DESCEND vers la zone

        2. EN HAUT À GAUCHE de la zone:
           → Se rapproche s'il va EN BAS ET À DROITE

        3. EN HAUT À DROITE de la zone:
           → Se rapproche s'il va EN BAS ET À GAUCHE

        4. EN BAS À GAUCHE de la zone (rare):
           → Se rapproche s'il va EN HAUT ET À DROITE

        5. EN BAS À DROITE de la zone (rare):
           → Se rapproche s'il va EN HAUT ET À GAUCHE

        6. JUSTE À GAUCHE (même altitude):
           → Se rapproche s'il va À DROITE

        7. JUSTE À DROITE (même altitude):
           → Se rapproche s'il va À GAUCHE

        Args:
            v: Le vaisseau

        Returns:
            True si le vaisseau se dirige vers la zone d'atterrissage
        """

        # CAS 1: Dans la zone horizontalement
        # → Doit simplement descendre
        if self.est_dans_la_zone(v):
            return self.va_en_bas(v)

        # CAS 2: En haut à gauche de la zone
        # → Doit aller en bas à droite (diagonale)
        if self.est_en_haut_de_la_zone(v) and self.est_a_gauche_de_la_zone(v):
            return self.va_en_bas(v) and self.va_a_droite(v)

        # CAS 3: En haut à droite de la zone
        # → Doit aller en bas à gauche (diagonale)
        if self.est_en_haut_de_la_zone(v) and self.est_a_droite_de_la_zone(v):
            return self.va_en_bas(v) and self.va_a_gauche(v)

        # CAS 4: En bas à gauche de la zone (situation rare)
        # → Doit remonter et aller à droite
        if self.est_en_bas_de_la_zone(v) and self.est_a_gauche_de_la_zone(v):
            return self.va_en_haut(v) and self.va_a_droite(v)

        # CAS 5: En bas à droite de la zone (situation rare)
        # → Doit remonter et aller à gauche
        if self.est_en_bas_de_la_zone(v) and self.est_a_droite_de_la_zone(v):
            return self.va_en_haut(v) and self.va_a_gauche(v)

        # CAS 6: Juste au-dessus de la zone (centré horizontalement)
        # → Doit simplement descendre
        if self.est_en_haut_de_la_zone(v) and self.est_dans_la_zone(v):
            return self.va_en_bas(v)

        # CAS 7: À gauche, même altitude que la zone
        # → Doit aller à droite
        if self.est_a_gauche_de_la_zone(v) and \
           not self.est_en_haut_de_la_zone(v) and \
           not self.est_en_bas_de_la_zone(v):
            return self.va_a_droite(v)

        # CAS 8: À droite, même altitude que la zone
        # → Doit aller à gauche
        if self.est_a_droite_de_la_zone(v) and \
           not self.est_en_haut_de_la_zone(v) and \
           not self.est_en_bas_de_la_zone(v):
            return self.va_a_gauche(v)

        # Cas par défaut: position non gérée
        return False

    # ==========================================================================
    # UTILITAIRES
    # ==========================================================================

    def get_zone_center(self) -> Tuple[float, float]:
        """
        Retourne le centre de la zone d'atterrissage.

        Utile pour calculer la distance du vaisseau à la zone
        et pour l'affichage du mini-radar.

        Returns:
            Tuple (x, y) du point central de la zone
        """
        center_x = (self.atterissage[0][0] + self.atterissage[1][0]) / 2
        center_y = self.atterissage[0][1]
        return (center_x, center_y)

    def get_zone_width(self) -> float:
        """
        Retourne la largeur de la zone d'atterrissage.

        Plus la zone est large, plus il est facile d'atterrir.

        Returns:
            Largeur de la zone en pixels
        """
        return self.atterissage[1][0] - self.atterissage[0][0]
