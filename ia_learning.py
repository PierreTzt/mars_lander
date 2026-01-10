"""
Module d'apprentissage par renforcement (Q-Learning) pour Mars Lander.

Ce module implémente l'algorithme Q-Learning pour entraîner une IA
à faire atterrir un vaisseau spatial sur Mars de manière autonome.

CONCEPTS CLÉS DU Q-LEARNING:
============================

1. Q-TABLE:
   - Une table qui associe chaque paire (état, action) à une valeur Q
   - La valeur Q représente "la qualité" d'une action dans un état donné
   - Plus la valeur est haute, meilleure est l'action

2. ÉTATS:
   - Représentation discrétisée de la situation actuelle
   - Inclut: position, vitesse, angle, carburant, etc.
   - Utilise des "buckets" pour transformer des valeurs continues en discrètes

3. ACTIONS:
   - Combinaisons possibles de (angle, puissance)
   - 65 actions au total: 13 angles × 5 puissances

4. RÉCOMPENSES:
   - Signal de feedback pour guider l'apprentissage
   - Positives: atterrissage réussi, se rapprocher de la zone
   - Négatives: crash, s'éloigner de la zone

5. ÉQUATION DE BELLMAN:
   Q(s,a) = Q(s,a) + α * (r + γ * max(Q(s',a')) - Q(s,a))

   où:
   - s = état actuel
   - a = action effectuée
   - r = récompense reçue
   - s' = nouvel état
   - α = taux d'apprentissage
   - γ = facteur de discount

6. EPSILON-GREEDY:
   - Stratégie de choix d'action
   - Avec probabilité ε: action aléatoire (exploration)
   - Avec probabilité 1-ε: meilleure action connue (exploitation)

7. EXPERIENCE REPLAY:
   - Stocke les expériences passées dans un buffer
   - Réapprend sur des échantillons aléatoires
   - Améliore la stabilité de l'apprentissage
"""

import random
import os
import pickle
from collections import deque
from typing import Tuple, List, Dict, Optional, Any

import numpy as np

# Import des paramètres de configuration
from data import (
    alpha, gamma, epsilon, epsilon_decay,  # Hyperparamètres Q-Learning
    charger_historique, vider_historique,  # Gestion de l'historique
    max_h_speed, max_v_speed, fenX, fenY,  # Paramètres du jeu
    bonus_fuel_actif, bonus_fuel_mult      # Bonus carburant
)


class IALearning:
    """
    Agent Q-Learning pour le contrôle du Mars Lander.

    Cette classe implémente l'algorithme Q-Learning pour apprendre
    à atterrir un vaisseau de manière optimale par essai-erreur.

    FLUX D'APPRENTISSAGE:
    1. Observer l'état actuel
    2. Choisir une action (exploration ou exploitation)
    3. Exécuter l'action
    4. Observer la récompense et le nouvel état
    5. Mettre à jour la Q-table
    6. Répéter

    Attributes:
        q_table (Dict): Table des valeurs Q - cœur de l'apprentissage
        epsilon (float): Taux d'exploration (0-1)
        alpha (float): Taux d'apprentissage (0-1)
        gamma (float): Facteur de discount pour les récompenses futures
        replay_buffer (deque): Buffer pour Experience Replay
        toutes_les_actions (List): Liste de toutes les actions possibles
    """

    # ==========================================================================
    # CONSTANTES DE RÉCOMPENSES
    # ==========================================================================
    # Ces valeurs définissent les signaux de feedback pour l'apprentissage

    REWARD_LANDING_SUCCESS = 100   # Grande récompense pour un atterrissage réussi
    REWARD_CRASH = -50             # Pénalité importante pour un crash
    REWARD_APPROACH = 2            # Bonus pour se rapprocher de la zone
    REWARD_MOVE_AWAY = -1          # Pénalité pour s'éloigner de la zone
    REWARD_IN_ZONE = 5             # Bonus pour être au-dessus de la zone
    REWARD_SPEED_CONTROL = 3       # Bonus pour une vitesse contrôlée
    REWARD_CORRECT_ANGLE = 2       # Bonus pour une orientation correcte
    REWARD_FUEL_BONUS = 0.05       # Bonus par unité de carburant restant (si bonus_fuel_actif)

    def __init__(self, scenar: dict, toutes_les_actions: List[Tuple],
                 alpha: float, gamma: float, epsilon: float,
                 epsilon_decay: float, ia_active: bool):
        """
        Initialise l'agent Q-Learning avec ses hyperparamètres.

        Args:
            scenar: Configuration du scénario de jeu
            toutes_les_actions: Liste des 65 actions possibles [(angle, puissance), ...]
            alpha: Taux d'apprentissage (0.1 recommandé)
                   - Contrôle la vitesse de mise à jour de la Q-table
            gamma: Facteur de discount (0.9 recommandé)
                   - Importance des récompenses futures vs immédiates
            epsilon: Taux d'exploration initial (0.8 recommandé)
                   - Probabilité de choisir une action aléatoire
            epsilon_decay: Facteur de décroissance (0.99991 recommandé)
                   - Réduit progressivement l'exploration
            ia_active: True si l'IA est activée
        """
        # Configuration
        self.scenar = scenar
        self.ia_active = ia_active

        # Hyperparamètres de l'algorithme
        self.alpha = alpha              # Learning rate
        self.gamma = gamma              # Discount factor
        self.epsilon = epsilon          # Exploration rate
        self.epsilon_decay = epsilon_decay

        # Q-TABLE: Le cœur de l'apprentissage
        # Structure: {état -> numpy.array des valeurs Q pour chaque action}
        # Exemple: {(0, 1, 2, ...) -> [0.5, -0.3, 1.2, ...]}
        self.q_table: Dict[Tuple, np.ndarray] = {}

        # Suivi des récompenses
        self.recompense = 0                    # Récompense de la frame actuelle
        self.recompenses_cumulees: List[float] = []  # Historique cumulé

        # Actions disponibles
        self.toutes_les_actions = toutes_les_actions

        # Cache pour conversion action → index (optimisation de performance)
        # Permet de trouver rapidement l'index d'une action dans la Q-table
        self.action_to_index = {action: i for i, action in enumerate(toutes_les_actions)}

        # =======================================================================
        # EXPERIENCE REPLAY
        # =======================================================================
        # Technique qui améliore la stabilité de l'apprentissage en:
        # 1. Stockant les expériences passées (état, action, récompense, ...)
        # 2. Réapprenant sur des échantillons aléatoires de ces expériences
        #
        # Avantages:
        # - Brise la corrélation entre expériences consécutives
        # - Permet de réutiliser les expériences rares (atterrissages réussis)

        self.replay_buffer: deque = deque(maxlen=10000)  # Buffer circulaire
        self.batch_size = 32          # Nombre d'expériences par batch
        self.min_replay_size = 100    # Taille minimum avant de commencer le replay

        # Statistiques de l'entraînement
        self.episodes_count = 0        # Nombre d'épisodes (tentatives)
        self.successful_landings = 0   # Nombre d'atterrissages réussis

        # Historique pour visualisation de l'apprentissage
        self.episode_rewards: List[float] = []  # Récompense totale par épisode
        self.episode_lengths: List[int] = []    # Durée de chaque épisode (frames)
        self.current_episode_reward = 0         # Cumul de l'épisode en cours
        self.current_episode_length = 0         # Durée de l'épisode en cours

    # ==========================================================================
    # GESTION DE L'HISTORIQUE (PERSISTANCE)
    # ==========================================================================

    def recupere_historique(self) -> None:
        """
        Charge la Q-table depuis le dernier fichier d'historique.

        Permet de reprendre l'apprentissage d'une session précédente
        au lieu de recommencer de zéro. La Q-table contient toutes
        les connaissances accumulées par l'agent.

        Les fichiers sont stockés dans: historique/*.pkl
        """
        if not self.ia_active:
            return

        # Charger seulement si demandé et si la Q-table est vide
        if charger_historique and len(self.q_table) == 0:
            # Créer le dossier s'il n'existe pas
            if not os.path.exists("historique"):
                os.makedirs("historique")
                return

            # Récupérer la liste des fichiers
            fichiers = os.listdir("historique")
            if len(fichiers) > 0:
                # Prendre le dernier fichier (le plus récent par nom)
                fichier_historique = fichiers[-1]
                try:
                    with open(f"historique/{fichier_historique}", "rb") as f:
                        self.q_table = pickle.load(f)
                    print(f"Q-table chargée: {len(self.q_table)} états")
                except (FileNotFoundError, pickle.PickleError) as e:
                    print(f"Erreur lors du chargement de l'historique: {e}")

    def supprimer_historique(self) -> None:
        """
        Supprime les fichiers d'historique existants.

        Appelé après le chargement pour éviter d'accumuler
        trop de fichiers dans le dossier historique.
        """
        if not self.ia_active:
            return

        if vider_historique:
            if not os.path.exists("historique"):
                return

            fichiers = os.listdir("historique")
            for f in fichiers:
                try:
                    os.remove(f"historique/{f}")
                except OSError as e:
                    print(f"Erreur lors de la suppression: {e}")

    # ==========================================================================
    # CHOIX D'ACTION (STRATÉGIE EPSILON-GREEDY)
    # ==========================================================================

    def choisir_action(self, etat: Tuple) -> Optional[Tuple]:
        """
        Choisit une action avec la stratégie epsilon-greedy.

        EPSILON-GREEDY:
        - Avec probabilité epsilon: choisir une action ALÉATOIRE (exploration)
          → Permet de découvrir de nouvelles stratégies
        - Avec probabilité 1-epsilon: choisir la MEILLEURE action (exploitation)
          → Utilise les connaissances acquises

        Au début: epsilon élevé = beaucoup d'exploration
        À la fin: epsilon faible = surtout exploitation

        Args:
            etat: L'état actuel (tuple de valeurs discrètes)

        Returns:
            Tuple (angle, puissance) ou None si IA inactive
        """
        if not self.ia_active:
            return None

        # Tirage aléatoire pour décider exploration vs exploitation
        if random.uniform(0, 1) < self.epsilon:
            # EXPLORATION: action aléatoire
            return random.choice(self.toutes_les_actions)
        else:
            # EXPLOITATION: meilleure action selon la Q-table
            return self.meilleure_action(etat)

    def meilleure_action(self, etat: Tuple) -> Optional[Tuple]:
        """
        Retourne la meilleure action pour un état donné.

        Consulte la Q-table pour trouver l'action avec
        la plus grande valeur Q dans cet état.

        Args:
            etat: L'état actuel

        Returns:
            L'action (angle, puissance) avec la valeur Q maximale
        """
        if not self.ia_active:
            return None

        # Initialiser l'état s'il n'existe pas dans la Q-table
        if etat not in self.q_table:
            # Toutes les actions commencent avec une valeur Q de 0
            self.q_table[etat] = np.zeros(len(self.toutes_les_actions))

        # Récupérer les valeurs Q pour toutes les actions
        q_values = self.q_table[etat]
        max_q = np.max(q_values)

        # Si plusieurs actions ont la même valeur max, en choisir une au hasard
        # (évite les biais vers les premières actions)
        best_indices = np.where(q_values == max_q)[0]
        best_index = random.choice(best_indices)

        return self.toutes_les_actions[best_index]

    # ==========================================================================
    # EXPERIENCE REPLAY
    # ==========================================================================

    def store_experience(self, etat: Tuple, action: Tuple, recompense: float,
                         next_etat: Tuple, is_terminal: bool) -> None:
        """
        Stocke une expérience dans le buffer de replay.

        Une "expérience" est un tuple contenant:
        - L'état avant l'action
        - L'action effectuée
        - La récompense reçue
        - L'état résultant
        - Si c'est un état terminal (fin d'épisode)

        Le buffer est circulaire: les anciennes expériences sont
        automatiquement supprimées quand la capacité est atteinte.

        Args:
            etat: État avant l'action
            action: Action effectuée (angle, puissance)
            recompense: Récompense reçue
            next_etat: État après l'action
            is_terminal: True si crash ou atterrissage
        """
        if self.ia_active:
            self.replay_buffer.append((etat, action, recompense, next_etat, is_terminal))

    def train_on_batch(self) -> None:
        """
        Entraîne l'agent sur un batch d'expériences aléatoires.

        EXPERIENCE REPLAY EN ACTION:
        1. Échantillonne aléatoirement 32 expériences du buffer
        2. Met à jour la Q-table pour chacune
        3. Cela "mixe" les expériences et améliore la convergence

        Ne commence l'entraînement que quand suffisamment
        d'expériences ont été collectées (min_replay_size).
        """
        if not self.ia_active:
            return

        # Attendre d'avoir assez d'expériences
        if len(self.replay_buffer) < self.min_replay_size:
            return

        # Échantillonner un batch aléatoire
        batch_size = min(self.batch_size, len(self.replay_buffer))
        batch = random.sample(self.replay_buffer, batch_size)

        # Mettre à jour la Q-table pour chaque expérience du batch
        for etat, action, recompense, next_etat, is_terminal in batch:
            self.update_q_table(etat, action, recompense, next_etat, is_terminal)

    # ==========================================================================
    # MISE À JOUR DE LA Q-TABLE (CŒUR DE L'ALGORITHME)
    # ==========================================================================

    def update_q_table(self, etat: Tuple, action: Tuple, recompense: float,
                       next_etat: Tuple, is_terminal: bool = False) -> None:
        """
        Met à jour la Q-table avec l'équation de Bellman.

        ÉQUATION DE BELLMAN (Q-LEARNING):
        Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]

        Explication:
        - Q(s,a): Valeur actuelle de l'action a dans l'état s
        - r: Récompense reçue
        - γ * max(Q(s',a')): Meilleure valeur Q future, avec discount
        - α: Taux d'apprentissage (force de la mise à jour)

        Le terme entre crochets est l'"erreur TD" (Temporal Difference):
        - Si positif: l'action était meilleure que prévu → augmenter Q
        - Si négatif: l'action était pire que prévu → diminuer Q

        Args:
            etat: État avant l'action
            action: Action effectuée
            recompense: Récompense reçue
            next_etat: État après l'action
            is_terminal: True si c'est un état final (pas de futur)
        """
        if not self.ia_active:
            return

        # Initialiser les états dans la Q-table si nécessaire
        if etat not in self.q_table:
            self.q_table[etat] = np.zeros(len(self.toutes_les_actions))
        if next_etat not in self.q_table:
            self.q_table[next_etat] = np.zeros(len(self.toutes_les_actions))

        # Trouver l'index de l'action dans la Q-table
        action_index = self.action_to_index.get(action)
        if action_index is None:
            return  # Action inconnue, ignorer

        # Calculer la meilleure valeur Q future
        if is_terminal:
            # État terminal: pas de futur, donc pas de récompense future
            best_future_q = 0
        else:
            # Prendre la meilleure valeur Q parmi toutes les actions possibles
            best_future_q = np.max(self.q_table[next_etat])

        # ÉQUATION DE MISE À JOUR Q-LEARNING
        # td_error = "ce que j'ai obtenu" - "ce que j'attendais"
        current_q = self.q_table[etat][action_index]
        td_error = recompense + self.gamma * best_future_q - current_q

        # Mettre à jour la valeur Q
        self.q_table[etat][action_index] += self.alpha * td_error

        # Limiter les valeurs Q pour éviter l'explosion numérique
        # (stabilité de l'entraînement)
        self.q_table[etat][action_index] = np.clip(
            self.q_table[etat][action_index], -200, 200
        )

    # ==========================================================================
    # REPRÉSENTATION DE L'ÉTAT (DISCRÉTISATION)
    # ==========================================================================

    def _bucket(self, value: float, boundaries: List[float]) -> int:
        """
        Discrétise une valeur continue en "bucket" (catégorie).

        Le Q-Learning nécessite des états discrets. Cette fonction
        convertit les valeurs continues (position, vitesse, etc.)
        en catégories numérotées.

        EXEMPLE:
        _bucket(150, [-100, 0, 100, 200]) → 2
        Car 150 est dans l'intervalle [100, 200[

        Args:
            value: Valeur continue à discrétiser
            boundaries: Liste des frontières de catégories (triées)

        Returns:
            Index du bucket (0, 1, 2, ... len(boundaries))
        """
        for i, boundary in enumerate(boundaries):
            if value < boundary:
                return i
        return len(boundaries)

    def recupere_etat(self, v, s) -> Optional[Tuple]:
        """
        Construit la représentation de l'état actuel.

        L'état est un tuple de valeurs discrètes qui capture
        les informations essentielles pour décider de l'action:

        COMPOSANTS DE L'ÉTAT:
        1. dx_bucket: Distance horizontale à la zone (7 buckets)
        2. dy_bucket: Distance verticale à la zone (7 buckets)
        3. h_speed_bucket: Vitesse horizontale (7 buckets)
        4. v_speed_bucket: Vitesse verticale (7 buckets)
        5. angle_bucket: Angle du vaisseau (7 buckets)
        6. fuel_bucket: Niveau de carburant (5 buckets)
        7. is_in_zone: Booléen - au-dessus de la zone?
        8. can_land: Booléen - conditions d'atterrissage OK?
        9. is_destroyed: Booléen - vaisseau détruit?
        10. is_landed: Booléen - vaisseau posé?

        Args:
            v: Le vaisseau (pour position, vitesse, angle, etc.)
            s: La surface (pour position de la zone d'atterrissage)

        Returns:
            Tuple représentant l'état, ou None si IA inactive
        """
        if not self.ia_active:
            return None

        # Calculer le centre de la zone d'atterrissage
        zone_center_x = (s.atterissage[0][0] + s.atterissage[1][0]) / 2
        zone_y = s.atterissage[0][1]

        # Distance au centre de la zone
        dx = v.x - zone_center_x  # Négatif = à gauche, Positif = à droite
        dy = v.y - zone_y         # Négatif = au-dessus, Positif = en-dessous

        # Discrétisation de la distance horizontale
        # Buckets: très à gauche, gauche, légèrement gauche, centre, légèrement droite, droite, très à droite
        dx_bucket = self._bucket(dx, [-2000, -500, -100, 0, 100, 500, 2000])

        # Discrétisation de la distance verticale
        dy_bucket = self._bucket(dy, [-1000, -500, -100, 0, 100, 500, 1000])

        # Discrétisation des vitesses
        h_speed_bucket = self._bucket(v.h_speed, [-40, -20, -5, 0, 5, 20, 40])
        v_speed_bucket = self._bucket(v.v_speed, [-50, -30, -10, 0, 10, 30, 50])

        # Discrétisation de l'angle
        # Négatif = penché à gauche, Positif = penché à droite
        angle_bucket = self._bucket(v.angle, [-60, -30, -10, 0, 10, 30, 60])

        # Discrétisation du carburant
        fuel_bucket = self._bucket(v.fuel, [0, 100, 300, 500, 800])

        # États booléens (directement utilisables)
        is_in_zone = s.est_dans_la_zone(v)   # Au-dessus de la zone?
        is_destroyed = v.detruit              # Vaisseau crashé?
        is_landed = v.est_pose                # Vaisseau posé?
        can_land = v.peut_atterir()           # Conditions d'atterrissage OK?

        # Construire le tuple d'état
        return (
            dx_bucket,       # Position horizontale relative
            dy_bucket,       # Position verticale relative
            h_speed_bucket,  # Vitesse horizontale
            v_speed_bucket,  # Vitesse verticale
            angle_bucket,    # Orientation
            fuel_bucket,     # Carburant restant
            is_in_zone,      # Dans la zone d'atterrissage?
            can_land,        # Peut atterrir en sécurité?
            is_destroyed,    # Crashé?
            is_landed        # Posé?
        )

    # ==========================================================================
    # SYSTÈME DE RÉCOMPENSES
    # ==========================================================================

    def recupere_recompense(self, a, v, s, j) -> float:
        """
        Calcule la récompense pour l'état actuel.

        Le système de récompenses guide l'apprentissage en indiquant
        à l'agent quels comportements sont souhaités.

        STRUCTURE DES RÉCOMPENSES:

        1. APPROCHE DE LA ZONE:
           - Se rapprocher: +2
           - S'éloigner: -1
           - Être dans la zone: +5
           - Descendre dans la zone: +2

        2. CONTRÔLE:
           - Vitesse horizontale OK: +3
           - Vitesse verticale OK: +3
           - Orientation correcte: +2 à +5
           - Peut atterrir: +10

        3. RÉSULTATS:
           - Atterrissage réussi: +100
           - Crash: -50

        Args:
            a: L'affichage (pour détecter les collisions)
            v: Le vaisseau
            s: La surface
            j: Le jeu

        Returns:
            La récompense totale pour cette frame
        """
        if not self.ia_active:
            return 0

        self.recompense = 0

        # --- RÉCOMPENSES POUR L'APPROCHE DE LA ZONE ---

        # Bonus/pénalité selon qu'on se rapproche ou s'éloigne
        if s.se_rapproche_de_la_zone(v):
            self.recompense += self.REWARD_APPROACH  # +2
        else:
            self.recompense += self.REWARD_MOVE_AWAY  # -1

        # Bonus supplémentaires quand on est au-dessus de la zone
        if s.est_dans_la_zone(v):
            self.recompense += self.REWARD_IN_ZONE  # +5

            # Bonus pour descendre vers la zone
            if s.va_en_bas(v):
                self.recompense += 2

            # Bonus pour être bien orienté
            if v.est_parfaitement_droit():
                self.recompense += 5  # Parfaitement horizontal
            elif v.est_droit():
                self.recompense += self.REWARD_CORRECT_ANGLE  # +2, presque horizontal

            # Gros bonus si toutes les conditions d'atterrissage sont réunies
            if v.peut_atterir():
                self.recompense += 10

        # --- RÉCOMPENSES POUR LE CONTRÔLE DE VITESSE ---

        # Vitesse horizontale sous contrôle?
        if not v.va_trop_vite_h():
            self.recompense += self.REWARD_SPEED_CONTROL  # +3
        else:
            self.recompense -= 2  # Pénalité

        # Vitesse verticale sous contrôle?
        if not v.va_trop_vite_v():
            self.recompense += self.REWARD_SPEED_CONTROL  # +3
        else:
            self.recompense -= 2  # Pénalité

        # --- RÉCOMPENSES POUR COLLISION/ATTERRISSAGE ---

        # Vérifier si le vaisseau touche le sol
        if j.touche_mars(a, v, s):
            if v.est_pose and not v.detruit:
                # ATTERRISSAGE RÉUSSI!
                self.recompense += self.REWARD_LANDING_SUCCESS  # +100
                self.successful_landings += 1

                # BONUS CARBURANT: récompense supplémentaire pour économiser du fuel
                # Plus le vaisseau atterrit avec du carburant, plus le bonus est grand
                if bonus_fuel_actif:
                    fuel_bonus = v.fuel * bonus_fuel_mult
                    self.recompense += fuel_bonus
            elif v.detruit:
                # CRASH
                self.recompense += self.REWARD_CRASH  # -50

        # Pénalité supplémentaire pour destruction (hors zone, etc.)
        if v.detruit and not v.est_pose:
            self.recompense -= 10

        return self.recompense

    # ==========================================================================
    # UTILITAIRES
    # ==========================================================================

    def decay_epsilon(self) -> None:
        """
        Décroît le taux d'exploration epsilon.

        Au fil du temps, l'agent passe de l'exploration (actions aléatoires)
        à l'exploitation (meilleures actions connues).

        Formule: epsilon = epsilon * epsilon_decay
        Avec un minimum de 0.01 pour toujours garder un peu d'exploration.
        """
        if self.ia_active:
            self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

    def ajout_recompense_cumulative(self, recompense: float) -> None:
        """
        Ajoute la récompense au cumul historique.

        Permet de suivre l'évolution des récompenses au cours d'un épisode.

        Args:
            recompense: Récompense à ajouter
        """
        if self.ia_active:
            if not self.recompenses_cumulees:
                self.recompenses_cumulees.append(recompense)
            else:
                # Ajouter au cumul précédent
                self.recompenses_cumulees.append(
                    self.recompenses_cumulees[-1] + recompense
                )

    def recup_recompense(self) -> float:
        """
        Retourne la récompense cumulative actuelle.

        Returns:
            La somme de toutes les récompenses de l'épisode, ou 0
        """
        return self.recompenses_cumulees[-1] if self.recompenses_cumulees else 0

    def ia_appui(self, key: Any) -> Optional[Any]:
        """
        Retourne la touche si l'IA est active (utilitaire legacy).

        Args:
            key: La touche pressée

        Returns:
            La touche si l'IA est active, None sinon
        """
        if self.ia_active:
            return key
        return None

    def track_episode_step(self, reward: float) -> None:
        """
        Enregistre une étape de l'épisode en cours.

        Appelé à chaque frame pour accumuler les récompenses
        et compter la durée de l'épisode.

        Args:
            reward: Récompense obtenue à cette frame
        """
        if self.ia_active:
            self.current_episode_reward += reward
            self.current_episode_length += 1

    def end_episode(self) -> None:
        """
        Termine l'épisode en cours et enregistre les statistiques.

        Appelé quand le vaisseau atterrit ou crash. Sauvegarde
        les métriques pour la visualisation de l'apprentissage.
        """
        if self.ia_active:
            # Enregistrer les métriques de l'épisode
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)

            # Limiter la taille de l'historique (garder les 1000 derniers)
            max_history = 1000
            if len(self.episode_rewards) > max_history:
                self.episode_rewards = self.episode_rewards[-max_history:]
                self.episode_lengths = self.episode_lengths[-max_history:]

            # Réinitialiser pour le prochain épisode
            self.current_episode_reward = 0
            self.current_episode_length = 0

    def get_recent_rewards(self, n: int = 100) -> List[float]:
        """
        Retourne les n dernières récompenses d'épisodes.

        Utilisé pour le graphique de visualisation.

        Args:
            n: Nombre de récompenses à retourner

        Returns:
            Liste des n dernières récompenses d'épisodes
        """
        return self.episode_rewards[-n:] if self.episode_rewards else []

    def get_average_reward(self, n: int = 100) -> float:
        """
        Calcule la récompense moyenne sur les n derniers épisodes.

        Args:
            n: Nombre d'épisodes à considérer

        Returns:
            Récompense moyenne
        """
        recent = self.get_recent_rewards(n)
        return sum(recent) / len(recent) if recent else 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques détaillées de l'agent.

        Returns:
            Dictionnaire contenant:
            - q_table_size: Nombre d'états connus
            - epsilon: Taux d'exploration actuel
            - replay_buffer_size: Expériences stockées
            - episodes: Nombre d'épisodes joués
            - successful_landings: Atterrissages réussis
            - success_rate: Taux de réussite
            - avg_reward: Récompense moyenne récente
        """
        return {
            "q_table_size": len(self.q_table),
            "epsilon": self.epsilon,
            "replay_buffer_size": len(self.replay_buffer),
            "episodes": self.episodes_count,
            "successful_landings": self.successful_landings,
            "success_rate": self.successful_landings / max(1, self.episodes_count),
            "avg_reward": self.get_average_reward(100)
        }
