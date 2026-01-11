"""
Mars Lander - Jeu de simulation d'atterrissage avec IA Q-Learning.

Ce programme simule l'atterrissage d'un vaisseau spatial sur Mars,
avec un agent d'apprentissage par renforcement (Q-Learning) qui apprend
à atterrir de manière autonome.

FONCTIONNALITÉS PRINCIPALES:
- Simulation physique réaliste (gravité, propulsion, inertie)
- Intelligence Artificielle par Q-Learning avec Experience Replay
- Interface graphique avancée avec effets de particules
- Système de sauvegarde/chargement de l'apprentissage
- Contrôle de vitesse de simulation en temps réel

CONTRÔLES:
- ESPACE: Redémarrer la simulation
- Flèches gauche/droite: Rotation du vaisseau (mode manuel)
- Touches 1-5: Puissance des moteurs (mode manuel)
- P: Pause/Reprise
- +/-: Ajuster la vitesse de simulation

AUTEUR: Projet amélioré avec Q-Learning et graphismes avancés

NOTE: Ce fichier redirige vers la version PRO avec PyTorch DQN.
Pour utiliser l'ancienne version, commentez le bloc ci-dessous.
"""

# === REDIRECTION VERS LA VERSION PRO ===
if __name__ == "__main__":
    print("Redirection vers Mars Lander ULTIMATE PRO Edition...")
    from mars_lander_pro import main as pro_main
    pro_main()
    import sys
    sys.exit(0)
# === FIN REDIRECTION ===

import os
import datetime
import pickle
import random
import math
import time

import pygame

# Import des configurations et constantes depuis data.py
from data import (
    scenario0, scenario1, scenario2, scenario3, scenario4, scenario5,
    alpha, gamma, epsilon, epsilon_decay,  # Paramètres Q-Learning
    img_par_sec, ia_active, affiche_espion,  # Configuration d'exécution
    # Nouvelles fonctionnalités
    vent_actif, vent_force_max, vent_changement_freq,
    sons_actifs, volume_effets,
    replay_actif, max_replays
)

# Import des classes principales du jeu
from vaisseau import Vaisseau      # Gestion du vaisseau spatial
from jeu import Jeu                # Logique du jeu
from affichage import Affichage    # Rendu graphique
from surface import Surface        # Terrain de Mars
from ia_learning import IALearning # Agent d'apprentissage


# =============================================================================
# SYSTÈME DE VENT
# =============================================================================

class WindSystem:
    """
    Système de vent qui perturbe le vaisseau de manière réaliste.

    Le vent change progressivement de direction et de force,
    simulant des conditions météorologiques martiennes.

    Attributes:
        force_x, force_y (float): Force actuelle du vent
        target_x, target_y (float): Force cible (transition progressive)
        frame_count (int): Compteur pour le changement de direction
    """

    def __init__(self):
        """Initialise le système de vent."""
        self.force_x = 0.0
        self.force_y = 0.0
        self.target_x = 0.0
        self.target_y = 0.0
        self.frame_count = 0

    def update(self) -> None:
        """
        Met à jour le vent à chaque frame.

        Change la direction cible périodiquement et fait une
        transition progressive vers cette cible.
        """
        if not vent_actif:
            self.force_x = 0
            self.force_y = 0
            return

        self.frame_count += 1

        # Changer la direction cible périodiquement
        if self.frame_count >= vent_changement_freq:
            self.frame_count = 0
            # Nouvelle direction aléatoire
            self.target_x = random.uniform(-vent_force_max, vent_force_max)
            self.target_y = random.uniform(-vent_force_max * 0.3, vent_force_max * 0.3)

        # Transition progressive vers la cible (lissage)
        smoothing = 0.02
        self.force_x += (self.target_x - self.force_x) * smoothing
        self.force_y += (self.target_y - self.force_y) * smoothing

    def apply_to_vessel(self, v) -> None:
        """
        Applique la force du vent au vaisseau.

        Args:
            v: Le vaisseau à affecter
        """
        if vent_actif and not v.detruit and not v.est_pose:
            v.h_speed += self.force_x
            v.v_speed += self.force_y

    def get_force(self) -> tuple:
        """Retourne la force actuelle du vent."""
        return (self.force_x, self.force_y)


# =============================================================================
# SYSTÈME AUDIO
# =============================================================================

class SoundManager:
    """
    Gestionnaire des effets sonores du jeu.

    Gère les sons de propulsion, crash, atterrissage réussi, etc.
    Génère des sons procéduralement si les fichiers n'existent pas.

    Attributes:
        sounds (Dict): Dictionnaire des sons chargés
        channels (Dict): Canaux audio pour chaque type de son
    """

    def __init__(self):
        """Initialise le gestionnaire de sons."""
        self.sounds = {}
        self.initialized = False
        self.thrust_playing = False

        if sons_actifs:
            try:
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                self._generate_sounds()
                self.initialized = True
                print("Système audio initialisé")
            except pygame.error as e:
                print(f"Impossible d'initialiser l'audio: {e}")

    def _generate_sounds(self) -> None:
        """
        Génère des sons procéduralement.

        Crée des sons simples en utilisant pygame.mixer.Sound
        avec des tableaux numpy pour les formes d'onde.
        """
        try:
            import numpy as np

            sample_rate = 22050

            # Son de propulsion (bruit blanc filtré)
            duration = 0.5
            t = np.linspace(0, duration, int(sample_rate * duration))
            thrust_wave = np.random.uniform(-0.3, 0.3, len(t))
            # Filtrage simple (moyenne mobile)
            kernel_size = 50
            thrust_wave = np.convolve(thrust_wave, np.ones(kernel_size)/kernel_size, mode='same')
            thrust_wave = (thrust_wave * 32767).astype(np.int16)
            thrust_stereo = np.column_stack((thrust_wave, thrust_wave))
            self.sounds['thrust'] = pygame.mixer.Sound(thrust_stereo)
            self.sounds['thrust'].set_volume(volume_effets * 0.3)

            # Son d'explosion (bruit avec decay)
            duration = 1.0
            t = np.linspace(0, duration, int(sample_rate * duration))
            explosion_wave = np.random.uniform(-1, 1, len(t))
            envelope = np.exp(-t * 5)  # Decay exponentiel
            explosion_wave = explosion_wave * envelope
            explosion_wave = (explosion_wave * 32767 * 0.5).astype(np.int16)
            explosion_stereo = np.column_stack((explosion_wave, explosion_wave))
            self.sounds['explosion'] = pygame.mixer.Sound(explosion_stereo)
            self.sounds['explosion'].set_volume(volume_effets * 0.7)

            # Son de succès (ton montant)
            duration = 0.8
            t = np.linspace(0, duration, int(sample_rate * duration))
            freq_start, freq_end = 400, 800
            freq = np.linspace(freq_start, freq_end, len(t))
            success_wave = np.sin(2 * np.pi * freq * t / sample_rate * 100)
            envelope = np.exp(-t * 2)
            success_wave = success_wave * envelope
            success_wave = (success_wave * 32767 * 0.4).astype(np.int16)
            success_stereo = np.column_stack((success_wave, success_wave))
            self.sounds['success'] = pygame.mixer.Sound(success_stereo)
            self.sounds['success'].set_volume(volume_effets * 0.5)

        except ImportError:
            print("numpy non disponible - sons désactivés")
        except Exception as e:
            print(f"Erreur génération sons: {e}")

    def play_thrust(self, power: int) -> None:
        """Joue le son de propulsion selon la puissance."""
        if not self.initialized or 'thrust' not in self.sounds:
            return

        if power > 0:
            if not self.thrust_playing:
                self.sounds['thrust'].play(loops=-1)
                self.thrust_playing = True
            self.sounds['thrust'].set_volume(volume_effets * 0.1 * power)
        else:
            if self.thrust_playing:
                self.sounds['thrust'].stop()
                self.thrust_playing = False

    def play_explosion(self) -> None:
        """Joue le son d'explosion."""
        if self.initialized and 'explosion' in self.sounds:
            self.sounds['explosion'].play()

    def play_success(self) -> None:
        """Joue le son de succès."""
        if self.initialized and 'success' in self.sounds:
            self.sounds['success'].play()

    def stop_all(self) -> None:
        """Arrête tous les sons."""
        if self.initialized:
            pygame.mixer.stop()
            self.thrust_playing = False


# =============================================================================
# SYSTÈME DE REPLAY
# =============================================================================

class ReplaySystem:
    """
    Système d'enregistrement et de lecture des replays.

    Enregistre les trajectoires des meilleures performances
    pour pouvoir les rejouer plus tard.

    Attributes:
        recordings (List): Liste des enregistrements
        current_recording (List): Enregistrement en cours
        is_recording (bool): True si en cours d'enregistrement
    """

    def __init__(self):
        """Initialise le système de replay."""
        self.recordings = []  # Liste des replays sauvegardés
        self.current_recording = []  # États du replay en cours
        self.is_recording = False
        self.best_fuel_remaining = 0

    def start_recording(self) -> None:
        """Commence un nouvel enregistrement."""
        if replay_actif:
            self.current_recording = []
            self.is_recording = True

    def record_frame(self, v) -> None:
        """
        Enregistre l'état du vaisseau pour cette frame.

        Args:
            v: Le vaisseau
        """
        if self.is_recording and replay_actif:
            self.current_recording.append({
                'x': v.x,
                'y': v.y,
                'angle': v.angle,
                'h_speed': v.h_speed,
                'v_speed': v.v_speed,
                'puissance': v.puissance,
                'fuel': v.fuel
            })

    def end_recording(self, success: bool, fuel_remaining: float) -> None:
        """
        Termine l'enregistrement et sauvegarde si c'est un succès.

        Args:
            success: True si atterrissage réussi
            fuel_remaining: Carburant restant
        """
        if not replay_actif or not self.is_recording:
            return

        self.is_recording = False

        if success and len(self.current_recording) > 0:
            # Sauvegarder si c'est un bon replay
            if fuel_remaining > self.best_fuel_remaining or len(self.recordings) < max_replays:
                self.recordings.append({
                    'frames': self.current_recording,
                    'fuel': fuel_remaining
                })
                self.best_fuel_remaining = max(self.best_fuel_remaining, fuel_remaining)

                # Garder seulement les meilleurs replays
                self.recordings.sort(key=lambda x: x['fuel'], reverse=True)
                self.recordings = self.recordings[:max_replays]

        self.current_recording = []


def ensure_historique_dir() -> None:
    """
    S'assure que le dossier 'historique' existe pour sauvegarder les Q-tables.

    Ce dossier stocke les fichiers .pkl contenant l'apprentissage de l'IA,
    permettant de reprendre l'entraînement lors de sessions ultérieures.
    """
    if not os.path.exists("historique"):
        os.makedirs("historique")


def save_q_table(ia: IALearning) -> None:
    """
    Sauvegarde la Q-table de l'IA dans un fichier pickle horodaté.

    La Q-table contient toutes les valeurs Q apprises par l'agent,
    représentant la "mémoire" de l'IA sur les meilleures actions
    à prendre dans chaque état.

    Args:
        ia: L'agent IA contenant la Q-table à sauvegarder

    Format du fichier: historique/YYYY-MM-DD_HH-MM-SS.pkl
    """
    ensure_historique_dir()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"historique/{timestamp}.pkl"

    try:
        with open(filename, "wb") as f:
            pickle.dump(ia.q_table, f)
        print(f"Q-table sauvegardée: {filename}")
    except OSError as e:
        print(f"Erreur lors de la sauvegarde: {e}")


def main():
    """
    Fonction principale du jeu - Point d'entrée de l'application.

    Cette fonction:
    1. Initialise tous les composants du jeu
    2. Charge l'historique d'apprentissage si disponible
    3. Exécute la boucle principale du jeu
    4. Sauvegarde l'apprentissage à la fermeture
    """

    # =========================================================================
    # CONFIGURATION DU SCÉNARIO
    # =========================================================================
    # Sélection du scénario de jeu (terrain et position initiale)
    # Scénarios disponibles: scenario0 à scenario5 (difficulté croissante)
    scenar = scenario0

    # =========================================================================
    # INITIALISATION DES OBJETS DU JEU
    # =========================================================================

    # Création et initialisation du vaisseau spatial
    # Le vaisseau contient: position, vitesse, angle, carburant, puissance
    v = Vaisseau()
    v.init_vaisseau(scenar['vaisseau'])

    # Création de la surface de Mars et calcul de la zone d'atterrissage
    # La zone d'atterrissage est le segment horizontal le plus long
    s = Surface(scenar['surface_mars'])
    zone = s.calcul_zone_atterissage(scenar)

    # Initialisation du système d'affichage graphique
    # Comprend: fond étoilé, arrière-plan martien, HUD, particules
    a = Affichage()
    a.init_terrain(scenar['surface_mars'], zone)
    a.set_landing_zone(zone)

    # Création du gestionnaire de jeu (logique, états, statistiques)
    j = Jeu(scenar)

    # =========================================================================
    # INITIALISATION DE L'INTELLIGENCE ARTIFICIELLE
    # =========================================================================

    # Génération de toutes les actions possibles (combinaisons angle/puissance)
    # Actions: tuples (angle, puissance) où angle ∈ [-90°, 90°], puissance ∈ [0, 4]
    toutes_actions_possible = j.toutes_actions_possibles(v)

    # Création de l'agent Q-Learning avec les hyperparamètres
    # - alpha: taux d'apprentissage (vitesse d'apprentissage)
    # - gamma: facteur de discount (importance des récompenses futures)
    # - epsilon: taux d'exploration (probabilité d'action aléatoire)
    # - epsilon_decay: décroissance de l'exploration au fil du temps
    ia = IALearning(
        scenar,
        toutes_actions_possible,
        alpha,
        gamma,
        epsilon,
        epsilon_decay,
        ia_active
    )

    # Chargement de l'apprentissage précédent (si disponible)
    # Permet de continuer l'entraînement d'une session précédente
    ia.recupere_historique()
    ia.supprimer_historique()  # Nettoie les anciens fichiers après chargement

    # =========================================================================
    # INITIALISATION DES NOUVEAUX SYSTÈMES
    # =========================================================================

    # Système de vent (perturbations météorologiques)
    wind = WindSystem()

    # Système audio (effets sonores)
    sound = SoundManager()

    # Système de replay (enregistrement des meilleures performances)
    replay = ReplaySystem()
    replay.start_recording()

    # Variables pour le tracking des sons
    last_destroyed_state = False
    last_landed_state = False

    # Timer pour afficher le vaisseau après atterrissage/crash
    terminal_time = None  # Moment où le vaisseau s'est posé/crashé
    LANDING_DELAY = 2.0   # Délai en secondes avant de relancer

    # Anti-rebond pour les touches de configuration
    s_key_released = True
    v_key_released = True
    t_key_released = True

    # =========================================================================
    # CONFIGURATION DE LA BOUCLE DE JEU
    # =========================================================================

    # Horloge Pygame pour contrôler le framerate
    clock = pygame.time.Clock()

    # Variables de contrôle de la boucle principale
    running = True
    pause_key_released = True  # Anti-rebond pour la touche pause

    # Système de contrôle de vitesse de simulation
    # Permet d'accélérer ou ralentir l'entraînement de l'IA
    speed_levels = [1, 5, 15, 30, 60, 120, 500, 1000, 5000, 10000]
    speed_index = speed_levels.index(60) if 60 in speed_levels else 4
    current_speed = speed_levels[speed_index]
    speed_key_released = True  # Anti-rebond pour les touches de vitesse

    # Liste des scénarios disponibles
    scenarios = [scenario0, scenario1, scenario2, scenario3, scenario4, scenario5]
    scenario_names = ["Facile", "Moyen", "Difficile", "Difficile+", "Très Difficile", "Expert"]
    current_scenario_index = 0

    # Affichage des instructions dans la console
    print("=== Mars Lander IA ===")
    print(f"Mode: {'IA Active' if ia_active else 'Manuel'}")
    print("Contrôles:")
    print("  ESPACE: Redémarrer")
    print("  Flèches: Rotation")
    print("  1-5: Puissance")
    print("  P: Pause")
    print("  +/-: Vitesse simulation")
    print("  F1-F6: Changer de scénario")
    print("  S: Sons on/off")
    print("  V: Vent on/off")
    print("  T: Trajectoire on/off")
    print("======================")

    # =========================================================================
    # BOUCLE PRINCIPALE DU JEU
    # =========================================================================
    while running:
        # Limitation du framerate selon la vitesse choisie
        clock.tick(current_speed)

        # --- GESTION DES ÉVÉNEMENTS PYGAME ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not running:
            break

        # --- CYCLE D'APPRENTISSAGE Q-LEARNING ---

        # 1. Observer l'état actuel du vaisseau
        # L'état est une représentation discrétisée de la situation
        etat = ia.recupere_etat(v, s)

        # 2. Choisir une action (exploration vs exploitation)
        # Exploration: action aléatoire pour découvrir
        # Exploitation: meilleure action connue selon la Q-table
        ia_action = ia.choisir_action(etat)

        # --- GESTION DES ENTRÉES CLAVIER ---
        keys = pygame.key.get_pressed()

        # Gestion de la pause (avec anti-rebond pour éviter le spam)
        if keys[pygame.K_p]:
            if pause_key_released:
                j.toggle_pause()
                pause_key_released = False
        else:
            pause_key_released = True

        # Gestion de la vitesse de simulation (+ et -)
        if keys[pygame.K_PLUS] or keys[pygame.K_KP_PLUS] or keys[pygame.K_EQUALS]:
            if speed_key_released and speed_index < len(speed_levels) - 1:
                speed_index += 1
                current_speed = speed_levels[speed_index]
                speed_key_released = False
        elif keys[pygame.K_MINUS] or keys[pygame.K_KP_MINUS]:
            if speed_key_released and speed_index > 0:
                speed_index -= 1
                current_speed = speed_levels[speed_index]
                speed_key_released = False
        else:
            speed_key_released = True

        # Transmettre la vitesse actuelle à l'affichage pour le HUD
        a.current_speed = current_speed

        # --- CHANGEMENT DE SCÉNARIO (F1-F6) ---
        scenario_keys = [pygame.K_F1, pygame.K_F2, pygame.K_F3, pygame.K_F4, pygame.K_F5, pygame.K_F6]
        for i, key in enumerate(scenario_keys):
            if keys[key] and i != current_scenario_index:
                current_scenario_index = i
                scenar = scenarios[i]

                # Réinitialiser avec le nouveau scénario
                v = Vaisseau()
                v.init_vaisseau(scenar['vaisseau'])
                s = Surface(scenar['surface_mars'])
                zone = s.calcul_zone_atterissage(scenar)
                a.init_terrain(scenar['surface_mars'], zone)
                a.set_landing_zone(zone)
                j.scenar = scenar
                j.tentative = 0
                j.att_reussi = 0

                # Reset des états
                last_destroyed_state = False
                last_landed_state = False
                terminal_time = None  # Réinitialiser le timer
                replay.start_recording()

                print(f"Scénario changé: {scenario_names[i]}")
                break

        # --- TOGGLES CONFIGURATION ---
        # S: Toggle sons
        if keys[pygame.K_s]:
            if s_key_released:
                sound.initialized = not sound.initialized
                if not sound.initialized:
                    sound.stop_all()
                print(f"Sons: {'ON' if sound.initialized else 'OFF'}")
                s_key_released = False
        else:
            s_key_released = True

        # V: Toggle vent
        if keys[pygame.K_v]:
            if v_key_released:
                import data
                data.vent_actif = not data.vent_actif
                print(f"Vent: {'ON' if data.vent_actif else 'OFF'}")
                v_key_released = False
        else:
            v_key_released = True

        # T: Toggle trajectoire
        if keys[pygame.K_t]:
            if t_key_released:
                import data
                data.trajectoire_active = not data.trajectoire_active
                print(f"Trajectoire: {'ON' if data.trajectoire_active else 'OFF'}")
                t_key_released = False
        else:
            t_key_released = True

        # Relancer le jeu manuellement avec ESPACE
        if keys[pygame.K_SPACE]:
            ia.end_episode()  # Enregistrer les stats
            v = j.je_relance_le_jeu(v)
            ia.episodes_count += 1
            last_destroyed_state = False
            last_landed_state = False
            terminal_time = None  # Réinitialiser le timer
            replay.start_recording()

        # --- CONTRÔLE DU VAISSEAU ---
        if not v.detruit and not v.est_pose and not j.paused:
            if not ia_active:
                # MODE MANUEL: Le joueur contrôle le vaisseau
                from data import degres_par_tour
                if keys[pygame.K_RIGHT]:
                    v.angle -= degres_par_tour  # Rotation droite
                if keys[pygame.K_LEFT]:
                    v.angle += degres_par_tour  # Rotation gauche
                # Touches 1-5 pour la puissance des moteurs
                if keys[pygame.K_1]:
                    v.puissance = 0
                if keys[pygame.K_2]:
                    v.puissance = 1
                if keys[pygame.K_3]:
                    v.puissance = 2
                if keys[pygame.K_4]:
                    v.puissance = 3
                if keys[pygame.K_5]:
                    v.puissance = 4
            else:
                # MODE IA: L'agent contrôle le vaisseau
                if ia_action is not None:
                    v.angle, v.puissance = ia_action

        # --- MISE À JOUR DE LA SIMULATION ---
        if not j.paused:
            # Mettre à jour et appliquer le vent
            wind.update()
            wind.apply_to_vessel(v)

            # Actualiser la physique du vaisseau (gravité, propulsion)
            j.actualisation(v, a, s, ia)

            # Enregistrer pour le replay
            replay.record_frame(v)

            # Gérer les sons de propulsion
            sound.play_thrust(v.puissance if not v.detruit and not v.est_pose else 0)

        # --- RENDU GRAPHIQUE ---
        # Dessiner tous les éléments: fond, terrain, vaisseau, HUD, particules
        j.affichage_du_jeu(a, v, s, ia)

        # Dessiner les éléments supplémentaires
        a.dessiner_trajectoire(v, wind.get_force())
        a.dessiner_indicateur_vent(wind.get_force())
        a.dessiner_graphique_apprentissage(ia)

        # Mettre à jour l'affichage
        pygame.display.flip()

        # --- DÉTECTION DES COLLISIONS ---
        # Vérifier si le vaisseau touche la surface de Mars
        j.touche_mars(a, v, s)

        # --- EFFETS SONORES ÉVÉNEMENTIELS ---
        # Jouer les sons de crash ou succès (une seule fois)
        if v.detruit and not last_destroyed_state:
            sound.play_explosion()
            sound.stop_all()  # Arrêter le son de propulsion
            replay.end_recording(False, 0)
            terminal_time = time.time()  # Enregistrer le moment du crash
        if v.est_pose and not last_landed_state:
            sound.play_success()
            sound.stop_all()
            replay.end_recording(True, v.fuel)
            terminal_time = time.time()  # Enregistrer le moment de l'atterrissage

        last_destroyed_state = v.detruit
        last_landed_state = v.est_pose

        # --- APPRENTISSAGE Q-LEARNING ---

        # 3. Calculer la récompense obtenue
        # Récompenses positives: se rapprocher de la zone, atterrir
        # Récompenses négatives: s'éloigner, crash, vitesse excessive
        recompense = ia.recupere_recompense(a, v, s, j)
        ia.ajout_recompense_cumulative(recompense)
        ia.track_episode_step(recompense)  # Tracking pour le graphique

        # 4. Observer le nouvel état après l'action
        next_etat = ia.recupere_etat(v, s)

        # Vérifier si c'est un état terminal (fin d'épisode)
        is_terminal = v.detruit or v.est_pose

        # 5. Mettre à jour la Q-table avec l'équation de Bellman
        # Q(s,a) = Q(s,a) + α * (r + γ * max(Q(s',a')) - Q(s,a))
        if etat is not None and ia_action is not None:
            ia.update_q_table(etat, ia_action, recompense, next_etat, is_terminal)

            # 6. Experience Replay: stocker et réapprendre des expériences passées
            # Améliore la stabilité et l'efficacité de l'apprentissage
            ia.store_experience(etat, ia_action, recompense, next_etat, is_terminal)
            ia.train_on_batch()

        # 7. Décroissance progressive de l'exploration (epsilon)
        # Au début: beaucoup d'exploration (découverte)
        # À la fin: plus d'exploitation (utilisation des connaissances)
        ia.decay_epsilon()

        # --- MODE DEBUG (ESPION) ---
        # Affiche des informations de débogage dans la console
        if affiche_espion:
            espion_parts = []
            if v.en_dehors_de_la_zone():
                espion_parts.append("HORS_ZONE")
            if v.peut_atterir():
                espion_parts.append("PEUT_ATTERRIR")
            if s.est_dans_la_zone(v):
                espion_parts.append("DANS_ZONE")
            if s.se_rapproche_de_la_zone(v):
                espion_parts.append("APPROCHE")
            if v.detruit:
                espion_parts.append("DETRUIT")
            if v.est_pose:
                espion_parts.append("POSE")
            if espion_parts:
                print(" | ".join(espion_parts))

        # --- RELANCE AUTOMATIQUE EN MODE IA ---
        # Quand un épisode se termine, attendre un délai puis recommencer
        if is_terminal and ia_active:
            # Attendre le délai pour voir le vaisseau posé/crashé
            if terminal_time is not None and (time.time() - terminal_time) >= LANDING_DELAY:
                ia.end_episode()  # Enregistrer les stats de l'épisode
                ia.episodes_count += 1
                v = j.je_relance_le_jeu(v)

                # Réinitialiser pour le nouvel épisode
                last_destroyed_state = False
                last_landed_state = False
                terminal_time = None  # Réinitialiser le timer
                replay.start_recording()

                # Afficher les statistiques tous les 100 épisodes
                if ia.episodes_count % 100 == 0:
                    stats = ia.get_stats()
                    print(f"Episode {stats['episodes']} | "
                          f"Réussites: {stats['successful_landings']} | "
                          f"Taux: {stats['success_rate']:.2%} | "
                          f"Epsilon: {stats['epsilon']:.4f} | "
                          f"Q-Table: {stats['q_table_size']}")

    # =========================================================================
    # FIN DU JEU - NETTOYAGE ET SAUVEGARDE
    # =========================================================================

    # Arrêter les sons
    sound.stop_all()

    # Sauvegarder l'apprentissage avant de quitter
    save_q_table(ia)

    # Afficher le nombre de replays enregistrés
    if replay_actif:
        print(f"Replays enregistrés: {len(replay.recordings)}")

    pygame.quit()

    # Afficher les statistiques finales de la session
    print("\n=== Statistiques finales ===")
    stats = ia.get_stats()
    print(f"Episodes: {stats['episodes']}")
    print(f"Atterrissages réussis: {stats['successful_landings']}")
    print(f"Taux de réussite: {stats['success_rate']:.2%}")
    print(f"Taille Q-Table: {stats['q_table_size']}")
    print("=============================")


# Point d'entrée du programme
if __name__ == "__main__":
    main()
