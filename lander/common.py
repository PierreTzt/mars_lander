"""
Systèmes partagés entre les différentes versions du jeu : vent, sons et replays.

Les options (vent_actif, sons_actifs, ...) sont lues dans le module `data`
au moment de l'appel, pour que les bascules clavier prennent effet en jeu.
"""

import random

import pygame

from . import data

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
        if not data.vent_actif:
            self.force_x = 0
            self.force_y = 0
            return

        self.frame_count += 1

        # Changer la direction cible périodiquement
        if self.frame_count >= data.vent_changement_freq:
            self.frame_count = 0
            # Nouvelle direction aléatoire
            self.target_x = random.uniform(-data.vent_force_max, data.vent_force_max)
            self.target_y = random.uniform(-data.vent_force_max * 0.3, data.vent_force_max * 0.3)

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
        if data.vent_actif and not v.detruit and not v.est_pose:
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

        if data.sons_actifs:
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
            self.sounds['thrust'].set_volume(data.volume_effets * 0.3)

            # Son d'explosion (bruit avec decay)
            duration = 1.0
            t = np.linspace(0, duration, int(sample_rate * duration))
            explosion_wave = np.random.uniform(-1, 1, len(t))
            envelope = np.exp(-t * 5)  # Decay exponentiel
            explosion_wave = explosion_wave * envelope
            explosion_wave = (explosion_wave * 32767 * 0.5).astype(np.int16)
            explosion_stereo = np.column_stack((explosion_wave, explosion_wave))
            self.sounds['explosion'] = pygame.mixer.Sound(explosion_stereo)
            self.sounds['explosion'].set_volume(data.volume_effets * 0.7)

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
            self.sounds['success'].set_volume(data.volume_effets * 0.5)

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
            self.sounds['thrust'].set_volume(data.volume_effets * 0.1 * power)
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
        if data.replay_actif:
            self.current_recording = []
            self.is_recording = True

    def record_frame(self, v) -> None:
        """
        Enregistre l'état du vaisseau pour cette frame.

        Args:
            v: Le vaisseau
        """
        if self.is_recording and data.replay_actif:
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
        if not data.replay_actif or not self.is_recording:
            return

        self.is_recording = False

        if success and len(self.current_recording) > 0:
            # Sauvegarder si c'est un bon replay
            if fuel_remaining > self.best_fuel_remaining or len(self.recordings) < data.max_replays:
                self.recordings.append({
                    'frames': self.current_recording,
                    'fuel': fuel_remaining
                })
                self.best_fuel_remaining = max(self.best_fuel_remaining, fuel_remaining)

                # Garder seulement les meilleurs replays
                self.recordings.sort(key=lambda x: x['fuel'], reverse=True)
                self.recordings = self.recordings[:data.max_replays]

        self.current_recording = []
