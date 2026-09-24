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
"""

import time

import pygame

# Import des configurations et constantes depuis data.py
from lander import data
from lander.data import (
    scenario0, scenario1, scenario2, scenario3, scenario4, scenario5,
    alpha, gamma, epsilon, epsilon_decay,  # Paramètres Q-Learning
    ia_active, affiche_espion,  # Configuration d'exécution
)

# Import des classes principales du jeu
from lander.vaisseau import Vaisseau      # Gestion du vaisseau spatial
from lander.jeu import Jeu                # Logique du jeu
from lander.affichage import Affichage    # Rendu graphique
from lander.surface import Surface        # Terrain de Mars
from lander.ia_learning import IALearning # Agent d'apprentissage
from lander.common import WindSystem, SoundManager, ReplaySystem


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
                data.vent_actif = not data.vent_actif
                print(f"Vent: {'ON' if data.vent_actif else 'OFF'}")
                v_key_released = False
        else:
            v_key_released = True

        # T: Toggle trajectoire
        if keys[pygame.K_t]:
            if t_key_released:
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
                if keys[pygame.K_RIGHT]:
                    v.angle -= data.degres_par_tour  # Rotation droite
                if keys[pygame.K_LEFT]:
                    v.angle += data.degres_par_tour  # Rotation gauche
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
    ia.sauvegarde_historique()

    # Afficher le nombre de replays enregistrés
    if data.replay_actif:
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
