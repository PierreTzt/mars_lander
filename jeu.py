"""
Module de gestion du jeu Mars Lander.

Ce module contient la classe Jeu qui orchestre la logique principale:
- Gestion des états du jeu (en cours, pause, terminé)
- Détection des collisions avec la surface de Mars
- Gestion des entrées clavier
- Génération des actions possibles pour l'IA
- Statistiques de jeu (tentatives, réussites)

ARCHITECTURE:
La classe Jeu fait le lien entre:
- Le vaisseau (physique et état)
- La surface (terrain et zone d'atterrissage)
- L'affichage (rendu graphique)
- L'IA (apprentissage)
"""

import pygame
from typing import List, Tuple, Optional, Any

# Import des constantes de configuration
from data import (
    ia_active,          # Mode IA ou manuel
    degres_par_tour,    # Rotation par appui de touche (15°)
    angle_vaisseau_max, # Angle maximum (±90°)
    echelle             # Facteur d'échelle pour l'affichage
)
from vaisseau import Vaisseau


class Jeu:
    """
    Gère la logique du jeu Mars Lander.

    Cette classe est le "chef d'orchestre" du jeu. Elle coordonne
    tous les autres composants et maintient l'état global du jeu.

    Attributes:
        tentative (int): Nombre total de tentatives (épisodes)
        att_reussi (int): Nombre d'atterrissages réussis
        scenar (dict): Configuration du scénario actuel
        paused (bool): True si le jeu est en pause
        est_gagne (bool): True si la dernière tentative a réussi
        toutes_les_actions (List): Toutes les actions possibles pour l'IA
    """

    def __init__(self, scenar: dict):
        """
        Initialise le jeu avec un scénario donné.

        Args:
            scenar: Dictionnaire contenant:
                - 'surface_mars': Points définissant le terrain
                - 'vaisseau': État initial du vaisseau
        """
        # Compteurs de statistiques
        self.tentative: int = 0      # Nombre d'épisodes joués
        self.att_reussi: int = 0     # Nombre de succès

        # Configuration
        self.scenar = scenar         # Scénario de jeu actuel

        # États du jeu
        self.est_gagne: bool = False # Résultat de la dernière tentative
        self.paused: bool = False    # État de pause

        # Liste des actions pour l'IA
        # Chaque action est un tuple (angle, puissance)
        self.toutes_les_actions: List[Tuple[int, int]] = []

    def actualisation(self, v, a, s, ia) -> None:
        """
        Met à jour l'état du jeu à chaque frame.

        Cette méthode est appelée à chaque itération de la boucle de jeu.
        Elle met à jour la physique du vaisseau si le jeu n'est pas en pause.

        Args:
            v: Le vaisseau (Vaisseau) - sera mis à jour
            a: L'affichage (Affichage) - non utilisé ici
            s: La surface (Surface) - non utilisé ici
            ia: L'agent IA (IALearning) - non utilisé ici
        """
        if not self.paused:
            # Mettre à jour la physique du vaisseau
            # (position, vitesse selon gravité et propulsion)
            v.actualisation()

            # Vérifier si le vaisseau est hors limites ou sans carburant
            v.verif_si_HS()

    def affichage_du_jeu(self, a, v, s, ia) -> None:
        """
        Dessine tous les éléments du jeu à l'écran.

        Ordre de dessin (du fond vers l'avant):
        1. Effacer l'écran (fond étoilé + arrière-plan Mars)
        2. HUD et informations
        3. Vaisseau avec effets de particules
        4. Surface de Mars

        Args:
            a: L'affichage (Affichage)
            v: Le vaisseau (Vaisseau)
            s: La surface (Surface)
            ia: L'agent IA (IALearning)
        """
        # 1. Effacer et dessiner le fond
        a.effacer_tout()

        # 2. Dessiner le HUD (interface utilisateur)
        a.ecrire_info(v, ia, self)

        # 3. Dessiner le vaisseau (avec flammes et particules)
        a.dessiner_vaisseau(v, self)

        # 4. Dessiner la surface de Mars
        a.dessiner_surface(s.mars_surface)

        # 5. Mettre à jour l'affichage Pygame
        pygame.display.flip()

    def toggle_pause(self) -> None:
        """
        Bascule l'état de pause du jeu.

        Si le jeu est en cours → met en pause
        Si le jeu est en pause → reprend
        """
        self.paused = not self.paused

    def touche_mars(self, a, v, s) -> bool:
        """
        Vérifie si le vaisseau touche la surface de Mars.

        Cette méthode effectue la détection de collision entre
        le rectangle du vaisseau et les segments du terrain.

        LOGIQUE DE COLLISION:
        1. Convertir les points du terrain en segments à l'échelle écran
        2. Vérifier si le rectangle du vaisseau intersecte un segment
        3. Si collision:
           - Dans la zone + bonnes conditions → SUCCÈS
           - Sinon → CRASH

        CONDITIONS DE SUCCÈS:
        - Le vaisseau doit être dans la zone d'atterrissage
        - La vitesse verticale doit être ≤ 40 m/s
        - La vitesse horizontale doit être ≤ 20 m/s
        - L'angle doit être 0° (horizontal)

        Args:
            a: L'affichage (pour obtenir le rectangle de collision)
            v: Le vaisseau
            s: La surface

        Returns:
            True si le vaisseau a touché la surface, False sinon
        """
        # Pas de rectangle = pas de collision possible
        if a.rect is None:
            return False

        # Construire la liste des segments du terrain à l'échelle écran
        points = []
        for i in range(len(s.mars_surface) - 1):
            # Récupérer deux points consécutifs
            pt1, pt2 = s.mars_surface[i]      # Point courant (x, y)
            pt3, pt4 = s.mars_surface[i + 1]  # Point suivant (x, y)

            # Convertir à l'échelle de l'écran
            pt1 = pt1 // echelle  # x1
            pt2 = pt2 // echelle  # y1
            pt3 = pt3 // echelle  # x2
            pt4 = pt4 // echelle  # y2

            # Ajouter le segment ((x1, y1), (x2, y2))
            points.append(((pt1, pt2), (pt3, pt4)))

        # Vérifier si le rectangle du vaisseau intersecte un segment
        # clipline() retourne les points d'intersection s'il y en a
        atterissage = any(a.rect.clipline(*point) for point in points)

        if atterissage:
            # Collision détectée - déterminer si c'est un succès ou un crash

            # Conditions de succès:
            # 1. Dans la zone d'atterrissage
            # 2. Vitesses acceptables (peut_atterir vérifie v_speed, h_speed, angle)
            # 3. Vaisseau pas déjà détruit
            if s.est_dans_la_zone(v) and v.peut_atterir() and not v.detruit:
                # ATTERRISSAGE RÉUSSI
                self.est_gagne = True
                v.est_pose = True
                self.att_reussi += 1
            else:
                # CRASH
                self.est_gagne = False
                v.detruit = True

            # Arrêter le vaisseau
            self.fin_du_jeu(v)
            return True

        return False

    def fin_du_jeu(self, v) -> None:
        """
        Arrête le vaisseau quand le jeu est terminé.

        Met les vitesses à zéro pour "figer" le vaisseau
        après un atterrissage ou un crash.

        Args:
            v: Le vaisseau
        """
        if v.detruit or v.est_pose:
            v.v_speed = 0  # Arrêter le mouvement vertical
            v.h_speed = 0  # Arrêter le mouvement horizontal

    def je_relance_le_jeu(self, v) -> Vaisseau:
        """
        Relance une nouvelle tentative (nouvel épisode).

        Crée un nouveau vaisseau avec les paramètres initiaux
        du scénario actuel. L'ancien vaisseau est abandonné.

        Args:
            v: Le vaisseau actuel (sera remplacé)

        Returns:
            Un nouveau vaisseau initialisé aux conditions de départ
        """
        # Incrémenter le compteur de tentatives
        self.tentative += 1

        # Créer un nouveau vaisseau
        v = Vaisseau()
        v.init_vaisseau(self.scenar['vaisseau'])

        return v

    def actions_clavier(self, keys, v, ia_action: Optional[Tuple]) -> Vaisseau:
        """
        Gère les entrées clavier du joueur.

        Note: Cette méthode est définie mais la gestion clavier
        est principalement faite dans mars_lander.py.

        Args:
            keys: État des touches du clavier (pygame.key.get_pressed())
            v: Le vaisseau
            ia_action: L'action choisie par l'IA (tuple) ou None

        Returns:
            Le vaisseau (potentiellement nouveau si relancé)
        """
        # Touche Espace : relancer le jeu
        if keys[pygame.K_SPACE]:
            v = self.je_relance_le_jeu(v)

        # Touche P : basculer la pause
        if keys[pygame.K_p]:
            self.toggle_pause()

        # Contrôles du vaisseau (seulement si en vol)
        if not v.detruit and not v.est_pose and not self.paused:
            if not ia_active:
                # MODE MANUEL - Le joueur contrôle

                # Rotation avec les flèches
                if keys[pygame.K_RIGHT]:
                    v.angle -= degres_par_tour  # Tourner à droite
                if keys[pygame.K_LEFT]:
                    v.angle += degres_par_tour  # Tourner à gauche

                # Puissance avec les touches 1-5
                if keys[pygame.K_1]:
                    v.puissance = 0  # Moteurs éteints
                if keys[pygame.K_2]:
                    v.puissance = 1
                if keys[pygame.K_3]:
                    v.puissance = 2
                if keys[pygame.K_4]:
                    v.puissance = 3
                if keys[pygame.K_5]:
                    v.puissance = 4  # Puissance maximale
            else:
                # MODE IA - L'agent contrôle
                if ia_action is not None:
                    v.angle, v.puissance = ia_action

        return v

    def toutes_actions_possibles(self, v) -> List[Tuple[int, int]]:
        """
        Génère toutes les combinaisons d'actions possibles.

        Une action est définie par:
        - Un angle: de -90° à +90° par pas de 15° (13 valeurs)
        - Une puissance: de 0 à 4 (5 valeurs)

        Total: 13 × 5 = 65 actions possibles

        Args:
            v: Le vaisseau (non utilisé, conservé pour compatibilité)

        Returns:
            Liste de tuples (angle, puissance)
        """
        self.toutes_les_actions = []

        # Parcourir toutes les puissances (0 à 4)
        for puissance in range(5):
            # Parcourir tous les angles (-90° à +90° par pas de 15°)
            for angle_mult in range(-6, 7):  # -6 à 6 inclus = 13 valeurs
                angle = angle_mult * degres_par_tour  # -90, -75, ..., 75, 90
                self.toutes_les_actions.append((angle, puissance))

        return self.toutes_les_actions

    def recup_actions_possibles(self, v) -> List[Tuple[int, int]]:
        """
        Filtre les actions possibles selon l'état actuel du vaisseau.

        CONTRAINTES PHYSIQUES RÉALISTES:
        - L'angle ne peut changer que de ±15° par tour
        - La puissance ne peut changer que de ±1 par tour

        Cette méthode simule les limitations mécaniques d'un vrai vaisseau
        qui ne peut pas changer instantanément d'orientation ou de poussée.

        Args:
            v: Le vaisseau (pour connaître l'angle et la puissance actuels)

        Returns:
            Liste des actions réalisables depuis l'état actuel
        """
        actions_possibles = []

        for action in self.toutes_les_actions:
            angle, puissance = action

            # Vérifier que l'angle cible est dans les limites globales
            if -angle_vaisseau_max <= angle <= angle_vaisseau_max and \
               0 <= puissance <= 4:

                # Vérifier les contraintes de changement progressif
                # L'angle peut changer de max 15° par tour
                delta_angle = abs(v.angle - angle)
                # La puissance peut changer de max 1 par tour
                delta_puissance = abs(puissance - v.puissance)

                if delta_angle <= degres_par_tour and delta_puissance <= 1:
                    actions_possibles.append(action)

        return actions_possibles

    def get_stats(self) -> dict:
        """
        Retourne les statistiques actuelles du jeu.

        Returns:
            Dictionnaire contenant:
            - 'tentatives': Nombre total d'épisodes
            - 'reussites': Nombre d'atterrissages réussis
            - 'taux_reussite': Ratio succès/tentatives
        """
        # Éviter la division par zéro
        success_rate = self.att_reussi / max(1, self.tentative)

        return {
            "tentatives": self.tentative,
            "reussites": self.att_reussi,
            "taux_reussite": success_rate
        }
