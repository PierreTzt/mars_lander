"""
Mars Lander V2.0 - Édition Complète avec 20 fonctionnalités.

Ce fichier intègre toutes les nouvelles fonctionnalités:
1. Météorites dynamiques
2. Mode nuit + éclairage
3. Plusieurs planètes
4. Dégâts progressifs
5. Zones d'atterrissage multiples
6. Deep Q-Network (DQN)
7. Caméra dynamique
8. Mode Time Attack
9. Tempêtes de poussière
10. Export GIF
11. Mode multijoueur local
12. Système de missions
13. Stations de ravitaillement
14. Algorithme génétique
15. Terrain destructible
16. Power-ups
17. Brouillard de guerre
18. Mode survie
19. Éditeur de niveaux
20. Dashboard statistiques

Contrôles:
- ESPACE: Redémarrer
- P: Pause
- +/-: Vitesse simulation
- F1-F6: Scénarios
- S: Sons on/off
- V: Vent on/off
- T: Trajectoire on/off
- N: Mode nuit on/off
- M: Météorites on/off
- B: Brouillard on/off
- G: Export GIF
- E: Éditeur de niveaux
- D: Dashboard statistiques
- C: Caméra dynamique on/off
- 1-6: Changer de planète
"""

import os
import sys
import time
import random
import math
import pygame

# Imports locaux
from data import *
import data as data_module  # Pour pouvoir modifier les valeurs
from vaisseau import Vaisseau
from jeu import Jeu
from affichage import Affichage
from surface import Surface
from ia_learning import IALearning

# Import des nouveaux systèmes
from game_systems import (
    MeteoriteSystem, DamageSystem, PowerUpSystem, FuelStationSystem,
    MultiZoneSystem, NightModeSystem, DustStormSystem, FogOfWarSystem,
    DynamicCamera, TimeAttackMode, SurvivalMode, MissionSystem
)
from advanced_ai import DQNAgent, GeneticAlgorithm, PlanetSystem
from extra_features import (
    DestructibleTerrain, MultiplayerMode, ScreenRecorder,
    LevelEditor, StatsDashboard
)

# Import du système de vent et sons de la version 1
from mars_lander import WindSystem, SoundManager, ReplaySystem


def print_help():
    """Affiche l'aide des contrôles."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║              MARS LANDER V2.0 - CONTRÔLES                    ║
╠══════════════════════════════════════════════════════════════╣
║  ESPACE    : Redémarrer le jeu                               ║
║  P         : Pause                                           ║
║  +/-       : Vitesse de simulation                           ║
║  F1-F6     : Changer de scénario                             ║
╠══════════════════════════════════════════════════════════════╣
║  S         : Sons on/off                                     ║
║  V         : Vent on/off                                     ║
║  T         : Trajectoire prédictive on/off                   ║
║  N         : Mode nuit on/off                                ║
║  M         : Météorites on/off                               ║
║  B         : Brouillard de guerre on/off                     ║
║  C         : Caméra dynamique on/off                         ║
║  O         : Tempête de poussière                            ║
║  U         : Power-ups on/off                                ║
╠══════════════════════════════════════════════════════════════╣
║  G         : Exporter en GIF                                 ║
║  E         : Éditeur de niveaux                              ║
║  D         : Dashboard statistiques                          ║
╠══════════════════════════════════════════════════════════════╣
║  F7        : Mode Time Attack on/off                         ║
║  F8        : Mode Survie on/off                              ║
║  F9        : Missions on/off                                 ║
╠══════════════════════════════════════════════════════════════╣
║  1-6       : Changer de planète (Lune/Mars/Terre/Europa...)  ║
╚══════════════════════════════════════════════════════════════╝
    """)


def main():
    """Point d'entrée principal du jeu."""

    # =========================================================================
    # INITIALISATION PYGAME
    # =========================================================================
    pygame.init()
    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

    # Dimensions de l'écran
    screen_width = fenX // echelle
    screen_height = fenY // echelle
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Mars Lander V2.0 - 20 Features Edition")

    # Police
    font = pygame.font.Font(None, 20)
    font_large = pygame.font.Font(None, 36)

    # =========================================================================
    # INITIALISATION DES COMPOSANTS DE BASE
    # =========================================================================
    scenar = scenario0
    v = Vaisseau()
    v.init_vaisseau(scenar['vaisseau'])
    s = Surface(scenar['surface_mars'])
    a = Affichage()
    j = Jeu(scenar)

    # Zone d'atterrissage
    zone = s.calcul_zone_atterissage(scenar)
    a.init_terrain(scenar['surface_mars'], zone)
    a.set_landing_zone(zone)

    # Actions possibles pour l'IA
    toutes_actions_possible = j.toutes_actions_possibles(v)

    # Agent Q-Learning
    ia = IALearning(
        scenar,
        toutes_actions_possible,
        alpha,
        gamma,
        epsilon,
        epsilon_decay,
        ia_active
    )

    # =========================================================================
    # INITIALISATION DES NOUVEAUX SYSTÈMES
    # =========================================================================

    # 1. Météorites
    meteorites = MeteoriteSystem(screen_width, screen_height)
    meteorites.enabled = meteorites_actif

    # 2. Mode nuit
    night_mode = NightModeSystem(screen_width, screen_height)
    night_mode.enabled = mode_nuit_actif

    # 3. Planètes
    planets = PlanetSystem()
    planets.set_planet(planete_actuelle)

    # 4. Dégâts progressifs
    damage_system = DamageSystem()
    damage_system.enabled = degats_progressifs_actif

    # 5. Zones multiples
    multi_zones = MultiZoneSystem()

    # 6. DQN
    dqn_agent = None
    if dqn_actif:
        dqn_agent = DQNAgent(
            state_size=8,
            action_size=65,
            hidden_sizes=dqn_hidden_sizes,
            learning_rate=dqn_learning_rate
        )

    # 7. Caméra dynamique
    camera = DynamicCamera(screen_width, screen_height)
    camera.enabled = camera_dynamique_actif

    # 8. Time Attack
    time_attack = TimeAttackMode()
    time_attack.enabled = time_attack_actif

    # 9. Tempêtes
    dust_storm = DustStormSystem(screen_width, screen_height)
    dust_storm.enabled = tempetes_actif

    # 10. GIF Export
    recorder = ScreenRecorder(gif_max_frames, gif_fps)

    # 11. Multijoueur
    multiplayer = MultiplayerMode()
    multiplayer.enabled = multijoueur_actif

    # 12. Missions
    missions = MissionSystem()
    missions.enabled = missions_actif
    if missions.enabled:
        missions.set_active_mission("first_landing")

    # 13. Stations fuel
    fuel_stations = FuelStationSystem()
    fuel_stations.enabled = stations_fuel_actif
    if fuel_stations.enabled:
        fuel_stations.add_station(screen_width // 2, screen_height // 3)

    # 14. Algorithme génétique
    genetic = None
    if genetique_actif:
        genetic = GeneticAlgorithm(
            population_size=genetique_population,
            mutation_rate=genetique_mutation_rate
        )

    # 15. Terrain destructible
    destructible_terrain = DestructibleTerrain(scenar['surface_mars'], echelle)
    destructible_terrain.enabled = terrain_destructible_actif

    # 16. Power-ups
    powerups = PowerUpSystem(screen_width, screen_height)
    powerups.enabled = powerups_actif

    # 17. Brouillard
    fog = FogOfWarSystem(screen_width, screen_height)
    fog.enabled = brouillard_actif

    # 18. Mode survie
    survival = SurvivalMode()
    survival.enabled = survie_actif

    # 19. Éditeur
    level_editor = LevelEditor(screen_width, screen_height, echelle)

    # 20. Dashboard
    dashboard = StatsDashboard(screen_width, screen_height)

    # Systèmes de la V1
    wind = WindSystem()
    sound = SoundManager()
    replay = ReplaySystem()
    replay.start_recording()

    # =========================================================================
    # VARIABLES DE CONTRÔLE
    # =========================================================================
    clock = pygame.time.Clock()
    running = True

    # Anti-rebond touches
    key_states = {k: True for k in [
        pygame.K_p, pygame.K_s, pygame.K_v, pygame.K_t, pygame.K_n,
        pygame.K_m, pygame.K_b, pygame.K_c, pygame.K_g, pygame.K_e,
        pygame.K_d, pygame.K_o, pygame.K_u, pygame.K_F7, pygame.K_F8, pygame.K_F9
    ]}

    # Vitesse
    speed_levels = [1, 5, 15, 30, 60, 120, 500, 1000]
    speed_index = 4
    current_speed = speed_levels[speed_index]
    speed_key_released = True

    # Scénarios
    scenarios = [scenario0, scenario1, scenario2, scenario3, scenario4, scenario5]
    scenario_names = ["Facile", "Moyen", "Difficile", "Difficile+", "Très Difficile", "Expert"]
    current_scenario_index = 0

    # États
    last_destroyed = False
    last_landed = False
    terminal_time = None
    LANDING_DELAY = 2.0
    episode_start_time = time.time()

    # Messages temporaires
    message = ""
    message_time = 0

    print_help()
    print(f"\nPlanète actuelle: {planets.get_current_planet().name}")
    print(f"Gravité: {planets.get_current_planet().gravity} m/s²")

    # =========================================================================
    # BOUCLE PRINCIPALE
    # =========================================================================
    while running:
        clock.tick(current_speed)
        dt = 1 / max(1, current_speed)

        # --- ÉVÉNEMENTS ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and level_editor.active:
                level_editor.handle_click(event.pos[0], event.pos[1], event.button)
            elif event.type == pygame.KEYDOWN and level_editor.active:
                msg = level_editor.handle_key(event.key)
                if msg:
                    message = msg
                    message_time = time.time()

        if not running:
            break

        keys = pygame.key.get_pressed()

        # --- GESTION DES TOUCHES ---

        # Pause
        if keys[pygame.K_p] and key_states[pygame.K_p]:
            j.toggle_pause()
            key_states[pygame.K_p] = False
        elif not keys[pygame.K_p]:
            key_states[pygame.K_p] = True

        # Vitesse
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

        # Toggle sons (S)
        if keys[pygame.K_s] and key_states[pygame.K_s]:
            sound.initialized = not sound.initialized
            message = f"Sons: {'ON' if sound.initialized else 'OFF'}"
            message_time = time.time()
            key_states[pygame.K_s] = False
        elif not keys[pygame.K_s]:
            key_states[pygame.K_s] = True

        # Toggle vent (V)
        if keys[pygame.K_v] and key_states[pygame.K_v]:
            data_module.data_module.vent_actif = not data_module.data_module.vent_actif
            message = f"Vent: {'ON' if data_module.data_module.vent_actif else 'OFF'}"
            message_time = time.time()
            key_states[pygame.K_v] = False
        elif not keys[pygame.K_v]:
            key_states[pygame.K_v] = True

        # Toggle trajectoire (T)
        if keys[pygame.K_t] and key_states[pygame.K_t]:
            data_module.trajectoire_active = not data_module.trajectoire_active
            message = f"Trajectoire: {'ON' if data_module.trajectoire_active else 'OFF'}"
            message_time = time.time()
            key_states[pygame.K_t] = False
        elif not keys[pygame.K_t]:
            key_states[pygame.K_t] = True

        # Toggle mode nuit (N)
        if keys[pygame.K_n] and key_states[pygame.K_n]:
            night_mode.enabled = not night_mode.enabled
            message = f"Mode nuit: {'ON' if night_mode.enabled else 'OFF'}"
            message_time = time.time()
            key_states[pygame.K_n] = False
        elif not keys[pygame.K_n]:
            key_states[pygame.K_n] = True

        # Toggle météorites (M)
        if keys[pygame.K_m] and key_states[pygame.K_m]:
            meteorites.enabled = not meteorites.enabled
            message = f"Météorites: {'ON' if meteorites.enabled else 'OFF'}"
            message_time = time.time()
            key_states[pygame.K_m] = False
        elif not keys[pygame.K_m]:
            key_states[pygame.K_m] = True

        # Toggle brouillard (B)
        if keys[pygame.K_b] and key_states[pygame.K_b]:
            fog.enabled = not fog.enabled
            if not fog.enabled:
                fog.reset()
            message = f"Brouillard: {'ON' if fog.enabled else 'OFF'}"
            message_time = time.time()
            key_states[pygame.K_b] = False
        elif not keys[pygame.K_b]:
            key_states[pygame.K_b] = True

        # Toggle caméra (C)
        if keys[pygame.K_c] and key_states[pygame.K_c]:
            camera.enabled = not camera.enabled
            if not camera.enabled:
                camera.reset()
            message = f"Caméra dynamique: {'ON' if camera.enabled else 'OFF'}"
            message_time = time.time()
            key_states[pygame.K_c] = False
        elif not keys[pygame.K_c]:
            key_states[pygame.K_c] = True

        # Export GIF (G)
        if keys[pygame.K_g] and key_states[pygame.K_g]:
            if recorder.recording:
                recorder.stop_recording()
                if recorder.export_gif("mars_landing.gif"):
                    message = "GIF exporté: mars_landing.gif"
                else:
                    message = "Export GIF échoué"
            else:
                recorder.start_recording()
                message = "Enregistrement GIF démarré..."
            message_time = time.time()
            key_states[pygame.K_g] = False
        elif not keys[pygame.K_g]:
            key_states[pygame.K_g] = True

        # Éditeur (E)
        if keys[pygame.K_e] and key_states[pygame.K_e]:
            if level_editor.active:
                level_editor.stop()
                message = "Éditeur fermé"
            else:
                level_editor.start()
                message = "Éditeur ouvert - Cliquez pour dessiner"
            message_time = time.time()
            key_states[pygame.K_e] = False
        elif not keys[pygame.K_e]:
            key_states[pygame.K_e] = True

        # Dashboard (D)
        if keys[pygame.K_d] and key_states[pygame.K_d]:
            dashboard.toggle()
            key_states[pygame.K_d] = False
        elif not keys[pygame.K_d]:
            key_states[pygame.K_d] = True

        # Tempête (O)
        if keys[pygame.K_o] and key_states[pygame.K_o]:
            dust_storm.enabled = True
            dust_storm.start_storm(tempete_duree, tempete_intensite)
            message = "Tempête déclenchée!"
            message_time = time.time()
            key_states[pygame.K_o] = False
        elif not keys[pygame.K_o]:
            key_states[pygame.K_o] = True

        # Power-ups (U)
        if keys[pygame.K_u] and key_states[pygame.K_u]:
            powerups.enabled = not powerups.enabled
            message = f"Power-ups: {'ON' if powerups.enabled else 'OFF'}"
            message_time = time.time()
            key_states[pygame.K_u] = False
        elif not keys[pygame.K_u]:
            key_states[pygame.K_u] = True

        # Time Attack (F7)
        if keys[pygame.K_F7] and key_states[pygame.K_F7]:
            time_attack.enabled = not time_attack.enabled
            if time_attack.enabled:
                time_attack.start()
            message = f"Time Attack: {'ON' if time_attack.enabled else 'OFF'}"
            message_time = time.time()
            key_states[pygame.K_F7] = False
        elif not keys[pygame.K_F7]:
            key_states[pygame.K_F7] = True

        # Survie (F8)
        if keys[pygame.K_F8] and key_states[pygame.K_F8]:
            survival.enabled = not survival.enabled
            if survival.enabled:
                survival.start()
            message = f"Mode Survie: {'ON' if survival.enabled else 'OFF'}"
            message_time = time.time()
            key_states[pygame.K_F8] = False
        elif not keys[pygame.K_F8]:
            key_states[pygame.K_F8] = True

        # Missions (F9)
        if keys[pygame.K_F9] and key_states[pygame.K_F9]:
            missions.enabled = not missions.enabled
            if missions.enabled:
                missions.set_active_mission("first_landing")
            message = f"Missions: {'ON' if missions.enabled else 'OFF'}"
            message_time = time.time()
            key_states[pygame.K_F9] = False
        elif not keys[pygame.K_F9]:
            key_states[pygame.K_F9] = True

        # Changement de planète (1-6)
        planet_keys = [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6]
        planet_names = ['moon', 'mars', 'earth', 'europa', 'titan', 'venus']
        for i, key in enumerate(planet_keys):
            if keys[key]:
                planets.set_planet(planet_names[i])
                planet = planets.get_current_planet()
                message = f"Planète: {planet.name} (g={planet.gravity})"
                message_time = time.time()
                break

        # Changement de scénario (F1-F6)
        scenario_keys = [pygame.K_F1, pygame.K_F2, pygame.K_F3, pygame.K_F4, pygame.K_F5, pygame.K_F6]
        for i, key in enumerate(scenario_keys):
            if keys[key] and i != current_scenario_index:
                current_scenario_index = i
                scenar = scenarios[i]
                v = Vaisseau()
                v.init_vaisseau(scenar['vaisseau'])
                s = Surface(scenar['surface_mars'])
                zone = s.calcul_zone_atterissage(scenar)
                a.init_terrain(scenar['surface_mars'], zone)
                a.set_landing_zone(zone)
                j.scenar = scenar
                j.tentative = 0
                j.att_reussi = 0
                last_destroyed = False
                last_landed = False
                terminal_time = None
                fog.reset()
                damage_system.reset()
                destructible_terrain = DestructibleTerrain(scenar['surface_mars'], echelle)
                message = f"Scénario: {scenario_names[i]}"
                message_time = time.time()
                if time_attack.enabled:
                    time_attack.start()
                break

        # Relance manuelle (ESPACE)
        if keys[pygame.K_SPACE]:
            ia.end_episode()
            v = j.je_relance_le_jeu(v)
            ia.episodes_count += 1
            last_destroyed = False
            last_landed = False
            terminal_time = None
            damage_system.reset()
            fog.reset()
            replay.start_recording()
            episode_start_time = time.time()
            if time_attack.enabled:
                time_attack.start()

        # --- SI ÉDITEUR ACTIF ---
        if level_editor.active:
            level_editor.draw(screen, font)
            pygame.display.flip()
            continue

        # --- SI DASHBOARD VISIBLE ---
        if dashboard.visible:
            a.effacer_tout()
            dashboard.draw(screen, font)
            pygame.display.flip()
            continue

        # --- LOGIQUE DU JEU ---
        if not v.detruit and not v.est_pose and not j.paused:
            # Choix de l'action IA
            etat = ia.recupere_etat(v, s)

            if dqn_actif and dqn_agent:
                ia_action = dqn_agent.choose_action(etat)
            elif genetique_actif and genetic:
                ia_action = genetic.choose_action(etat)
            else:
                ia_action = ia.choisir_action(etat)

            # Appliquer l'action
            if ia_active and ia_action:
                v.angle, v.puissance = ia_action

            # Contrôle manuel
            if not ia_active:
                if keys[pygame.K_RIGHT]:
                    v.angle -= degres_par_tour
                if keys[pygame.K_LEFT]:
                    v.angle += degres_par_tour

            # Mise à jour physique
            j.actualisation(v, a, s, ia)

            # Appliquer la gravité de la planète
            gravity_factor = planets.get_gravity_factor()
            v.v_speed += (gravity_factor - 1) * 0.1  # Ajustement de gravité

            # Vent
            if data_module.data_module.vent_actif:
                wind.update()
                wind_force = wind.get_force()
                v.h_speed += wind_force[0]
                v.v_speed += wind_force[1]

            # Tempête
            if dust_storm.enabled:
                dust_storm.update(dt)
                storm_wind = dust_storm.get_wind_effect()
                v.h_speed += storm_wind[0]
                v.v_speed += storm_wind[1]

            # Météorites
            if meteorites.enabled:
                meteorites.update()
                vessel_x = v.x / echelle
                vessel_y = (fenY - v.y) / echelle
                if meteorites.check_collision(vessel_x, vessel_y):
                    if damage_system.enabled:
                        _, destroyed = damage_system.take_damage(30, "meteorite")
                        if destroyed:
                            v.detruit = True
                    else:
                        v.detruit = True
                    message = "Touché par une météorite!"
                    message_time = time.time()

            # Power-ups
            if powerups.enabled:
                powerups.update(dt)
                vessel_x = v.x / echelle
                vessel_y = (fenY - v.y) / echelle
                collected = powerups.check_collection(vessel_x, vessel_y)
                if collected:
                    msg = powerups.apply_effect(collected, v, damage_system)
                    message = f"Power-up: {msg}"
                    message_time = time.time()
                    missions.check_objective("collect_powerups")

            # Stations fuel
            if fuel_stations.enabled:
                vessel_x = v.x / echelle
                vessel_y = (fenY - v.y) / echelle
                docked, fuel = fuel_stations.check_docking(vessel_x, vessel_y, v)

            # Brouillard
            if fog.enabled:
                vessel_x = v.x / echelle
                vessel_y = (fenY - v.y) / echelle
                fog.update(vessel_x, vessel_y)

            # Caméra
            if camera.enabled:
                vessel_x = v.x / echelle
                vessel_y = (fenY - v.y) / echelle
                camera.update(vessel_x, vessel_y, v.v_speed)

            # Son propulsion
            if sound.initialized:
                sound.play_thrust(v.puissance)

            # Enregistrement GIF
            if recorder.recording:
                recorder.capture_frame(screen)

        # --- COLLISION ---
        j.touche_mars(a, v, s)

        # Détection atterrissage/crash
        if v.detruit and not last_destroyed:
            sound.play_explosion()
            sound.stop_all()
            terminal_time = time.time()
            episode_duration = time.time() - episode_start_time

            # Terrain destructible
            if destructible_terrain.enabled:
                impact_speed = math.sqrt(v.h_speed**2 + v.v_speed**2)
                destructible_terrain.create_crater(v.x, fenY - v.y, impact_speed)

            # Mode survie
            if survival.enabled:
                if survival.on_crash():
                    message = "GAME OVER!"
                else:
                    message = f"Crash! Vies restantes: {survival.lives}"
                message_time = time.time()

            # Missions
            missions.on_crash()

            # Dashboard
            dashboard.record_episode(ia.recompense, episode_duration, False,
                                   (v.x, v.y))

        if v.est_pose and not last_landed:
            sound.play_success()
            sound.stop_all()
            terminal_time = time.time()
            episode_duration = time.time() - episode_start_time

            # Time Attack
            if time_attack.enabled:
                elapsed = time_attack.stop(True)
                message = f"Temps: {time_attack.format_time(elapsed)}"
                message_time = time.time()

            # Mode survie
            if survival.enabled:
                points = survival.on_landing(v.fuel, episode_duration)
                message = f"+{points} points! Niveau {survival.level}"
                message_time = time.time()

            # Missions
            missions.check_objective("land")
            missions.check_objective("land_with_fuel", {"fuel_percent": v.fuel / 10})
            missions.check_objective("land_in_time", {"time": episode_duration})

            # Dashboard
            dashboard.record_episode(ia.recompense, episode_duration, True,
                                   (v.x, v.y), v.fuel)

        last_destroyed = v.detruit
        last_landed = v.est_pose

        # --- APPRENTISSAGE ---
        if ia_active:
            recompense = ia.recupere_recompense(a, v, s, j)
            ia.ajout_recompense_cumulative(recompense)
            next_etat = ia.recupere_etat(v, s)
            is_terminal = v.detruit or v.est_pose

            if etat and ia_action:
                if dqn_actif and dqn_agent:
                    dqn_agent.remember(etat, ia_action, recompense, next_etat, is_terminal)
                    dqn_agent.replay()
                    dqn_agent.decay_epsilon()
                elif genetique_actif and genetic:
                    if is_terminal:
                        fitness = ia.recompense
                        genetic.set_fitness(fitness, v.est_pose)
                        if genetic.next_individual():
                            print(f"Génération {genetic.generation} terminée")
                else:
                    ia.update_q_table(etat, ia_action, recompense, next_etat, is_terminal)
                    ia.store_experience(etat, ia_action, recompense, next_etat, is_terminal)
                    ia.train_on_batch()

                ia.decay_epsilon()

            # Dashboard training stats
            dashboard.record_training_step(len(ia.q_table), ia.epsilon)

        # --- RELANCE AUTO ---
        is_terminal = v.detruit or v.est_pose
        if is_terminal and ia_active:
            if terminal_time and (time.time() - terminal_time) >= LANDING_DELAY:
                ia.end_episode()
                ia.episodes_count += 1
                v = j.je_relance_le_jeu(v)
                last_destroyed = False
                last_landed = False
                terminal_time = None
                damage_system.reset()
                fog.reset()
                episode_start_time = time.time()
                if time_attack.enabled:
                    time_attack.start()

                if ia.episodes_count % 100 == 0:
                    stats = ia.get_stats()
                    print(f"Episode {stats['episodes']} | "
                          f"Réussites: {stats['successful_landings']} | "
                          f"Taux: {stats['success_rate']:.2%}")

        # --- AFFICHAGE ---
        a.effacer_tout()
        a.ecrire_info(v, ia, j)

        # Trajectoire
        if data_module.trajectoire_active:
            vent_force = wind.get_force() if data_module.vent_actif else (0, 0)
            a.dessiner_trajectoire(v, vent_force)

        # Vaisseau
        a.dessiner_vaisseau(v, j)

        # Surface
        if destructible_terrain.enabled:
            modified_surface = destructible_terrain.get_modified_surface()
            # Note: Il faudrait redessiner avec la surface modifiée
            destructible_terrain.draw_craters(screen)
        a.dessiner_surface(s.mars_surface)

        # Graphique apprentissage
        if graphique_actif:
            a.dessiner_graphique_apprentissage(ia)

        # Indicateur vent
        if data_module.data_module.vent_actif:
            a.dessiner_indicateur_vent(wind.get_force())

        # Météorites
        if meteorites.enabled:
            meteorites.draw(screen)

        # Power-ups
        if powerups.enabled:
            powerups.draw(screen, font)

        # Stations fuel
        if fuel_stations.enabled:
            fuel_stations.draw(screen, font)

        # Tempête
        if dust_storm.enabled:
            dust_storm.draw(screen)

        # Brouillard
        if fog.enabled:
            fog.draw(screen)

        # Mode nuit
        if night_mode.enabled:
            vessel_x = v.x / echelle
            vessel_y = (fenY - v.y) / echelle
            night_mode.draw(screen, vessel_x, vessel_y, v.angle)

        # Barre de vie
        if damage_system.enabled:
            damage_system.draw_health_bar(screen, 10, screen_height - 60)

        # Time Attack
        if time_attack.enabled:
            time_attack.draw(screen, font, screen_width - 150, 10)

        # Survie
        if survival.enabled:
            survival.draw(screen, font, screen_width - 150, 80)

        # Missions
        if missions.enabled:
            missions.draw(screen, font, 10, screen_height - 120)

        # Info planète
        planets.draw_info(screen, font, screen_width - 150, screen_height - 50)

        # Message temporaire
        if message and time.time() - message_time < 3:
            msg_surface = font_large.render(message, True, (255, 255, 0))
            msg_rect = msg_surface.get_rect(center=(screen_width // 2, 50))
            screen.blit(msg_surface, msg_rect)

        # Indicateur enregistrement GIF
        if recorder.recording:
            rec_text = font.render(f"REC [{recorder.get_frame_count()}]", True, (255, 0, 0))
            screen.blit(rec_text, (screen_width - 80, screen_height - 20))

        pygame.display.flip()

    # =========================================================================
    # NETTOYAGE
    # =========================================================================
    sound.stop_all()

    # Sauvegarde
    if dqn_actif and dqn_agent:
        dqn_agent.save("dqn_model.pkl")
        print("Modèle DQN sauvegardé")
    if genetique_actif and genetic:
        genetic.save("genetic_population.pkl")
        print("Population génétique sauvegardée")

    pygame.quit()
    print("Merci d'avoir joué à Mars Lander V2.0!")


if __name__ == "__main__":
    main()
