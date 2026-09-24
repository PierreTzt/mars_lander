"""
Configuration et constantes pour Mars Lander.

Ce fichier centralise tous les paramètres configurables du jeu:
- Hyperparamètres de l'IA (Q-Learning)
- Paramètres d'affichage (dimensions, échelle)
- Palette de couleurs pour les graphismes
- Définition des scénarios de jeu

MODIFICATION DES PARAMÈTRES:
- Pour ajuster la difficulté: modifier les scénarios
- Pour ajuster l'apprentissage: modifier alpha, gamma, epsilon
- Pour ajuster les graphismes: modifier les couleurs
"""

# =============================================================================
# PARAMÈTRES IA (Q-Learning)
# =============================================================================
# Ces hyperparamètres contrôlent le comportement de l'apprentissage

# Taux d'apprentissage (learning rate)
# - Valeur entre 0 et 1
# - Plus élevé = apprentissage plus rapide mais moins stable
# - Plus bas = apprentissage plus lent mais plus stable
# - Recommandé: 0.1 à 0.3
alpha = 0.1

# Facteur de discount (discount factor)
# - Valeur entre 0 et 1
# - Plus élevé = l'agent valorise les récompenses futures
# - Plus bas = l'agent préfère les récompenses immédiates
# - Recommandé: 0.9 à 0.99
gamma = 0.9

# Taux d'exploration initial (epsilon)
# - Valeur entre 0 et 1
# - Probabilité de choisir une action aléatoire
# - 1.0 = 100% exploration (actions aléatoires)
# - 0.0 = 100% exploitation (meilleure action connue)
# - Recommandé: commencer à 0.8-1.0
epsilon = 0.8

# Facteur de décroissance de l'exploration
# - À chaque étape: epsilon = epsilon * epsilon_decay
# - Plus proche de 1 = décroissance lente (plus d'exploration)
# - Plus petit = décroissance rapide (moins d'exploration)
# - Exemple: 0.99991 réduit epsilon de moitié en ~7000 étapes
epsilon_decay = 0.99991

# =============================================================================
# NOUVELLES FONCTIONNALITÉS
# =============================================================================

# --- Effets météo (vent) ---
# Active/désactive le vent aléatoire qui perturbe le vaisseau
vent_actif = True
# Force maximale du vent (en pixels/frame²)
vent_force_max = 0.3
# Fréquence de changement du vent (en frames)
vent_changement_freq = 120

# --- Bonus carburant ---
# Bonus de récompense pour atterrir avec du carburant restant
bonus_fuel_actif = True
# Multiplicateur: bonus = fuel_restant * bonus_fuel_mult
bonus_fuel_mult = 0.05

# --- Trajectoire prédictive ---
# Affiche une ligne montrant où le vaisseau va atterrir
trajectoire_active = True
# Nombre de points de prédiction
trajectoire_points = 50

# --- Effets sonores ---
# Active/désactive les sons du jeu
sons_actifs = True
# Volume des effets (0.0 à 1.0)
volume_effets = 0.5

# --- Système de replay ---
# Enregistre les meilleures performances
replay_actif = True
# Nombre maximum de replays à conserver
max_replays = 10

# --- Visualisation de l'apprentissage ---
# Affiche un graphique des récompenses
graphique_actif = True
# Nombre de points à afficher dans le graphique
graphique_points = 100

# =============================================================================
# NOUVELLES FONCTIONNALITÉS (V2.0 - 20 Features)
# =============================================================================

# --- 1. Météorites dynamiques ---
meteorites_actif = False
meteorites_spawn_rate = 0.02  # Probabilité de spawn par frame
meteorites_difficulte = 1.0   # Multiplicateur de difficulté

# --- 2. Mode nuit et éclairage ---
mode_nuit_actif = False
nuit_obscurite = 200          # Niveau d'obscurité (0-255)
nuit_spot_radius = 150        # Rayon du spot lumineux

# --- 3. Planètes (gravités différentes) ---
# Options: 'mars', 'moon', 'earth', 'europa', 'titan', 'venus'
planete_actuelle = 'mars'

# --- 4. Dégâts progressifs ---
degats_progressifs_actif = False
degats_multiplicateur = 1.0

# --- 5. Zones d'atterrissage multiples ---
zones_multiples_actif = False  # Pas encore branché (voir MultiZoneSystem)

# --- 6. Deep Q-Network (DQN) ---
# Utiliser DQN au lieu de Q-Table classique
dqn_actif = False
dqn_hidden_sizes = [128, 64]  # Tailles des couches cachées
dqn_learning_rate = 0.001
dqn_batch_size = 32
dqn_memory_size = 10000

# --- 7. Caméra dynamique ---
camera_dynamique_actif = False
camera_auto_zoom = True
camera_zoom_min = 0.5
camera_zoom_max = 2.0

# --- 8. Mode Time Attack ---
time_attack_actif = False

# --- 9. Tempêtes de poussière ---
tempetes_actif = False
tempete_duree = 10.0          # Durée en secondes
tempete_intensite = 0.7       # Intensité (0-1)
tempete_probabilite = 0.05    # Probabilité de déclenchement

# --- 10. Export GIF ---
gif_export_actif = True
gif_max_frames = 500
gif_fps = 30

# --- 11. Mode multijoueur local ---
multijoueur_actif = False  # Pas encore branché : le système n'est ni mis à jour ni dessiné
multijoueur_rounds = 5

# --- 12. Système de missions ---
missions_actif = False

# --- 13. Stations de ravitaillement ---
stations_fuel_actif = False
stations_refuel_rate = 2.0    # Fuel par frame

# --- 14. Algorithme génétique ---
genetique_actif = False
genetique_population = 50
genetique_mutation_rate = 0.1

# --- 15. Terrain destructible ---
terrain_destructible_actif = False

# --- 16. Power-ups ---
powerups_actif = False
powerups_spawn_rate = 0.005

# --- 17. Brouillard de guerre ---
brouillard_actif = False
brouillard_reveal_radius = 100

# --- 18. Mode survie ---
survie_actif = False
survie_vies_max = 3
survie_landings_par_niveau = 3

# --- 19. Éditeur de niveaux ---
editeur_actif = True  # Toujours disponible avec touche E

# --- 20. Dashboard statistiques ---
dashboard_actif = True  # Toujours disponible avec touche D

# =============================================================================
# MODE DE FONCTIONNEMENT
# =============================================================================

# Active/désactive le mode IA
# - True: L'IA contrôle le vaisseau (entraînement automatique)
# - False: Mode manuel (le joueur contrôle)
ia_active = True

# Charger l'historique d'apprentissage au démarrage
# - True: Reprend l'apprentissage de la session précédente
# - False: Recommence l'apprentissage de zéro
charger_historique = True

# Supprimer les anciennes Q-tables après chargement (la plus récente est gardée)
# - True: Nettoie saves/qtable_*.pkl
# - False: Conserve tous les fichiers
vider_historique = True

# Configuration automatique selon le mode (IA ou manuel)
if ia_active:
    # MODE IA: Optimisé pour l'entraînement rapide
    affiche_espion = False   # Désactive les logs de debug
    img_par_sec = 10000      # Framerate élevé (simulation accélérée)
    gravite = 1              # Gravité standard
    angle_vaisseau_max = 90  # Rotation maximale (±90°)
else:
    # MODE MANUEL: Optimisé pour le jeu humain
    affiche_espion = True    # Active les logs de debug
    img_par_sec = 60         # 60 FPS (temps réel)
    gravite = 1              # Gravité standard
    angle_vaisseau_max = 90  # Rotation maximale (±90°)

# =============================================================================
# PARAMÈTRES D'AFFICHAGE
# =============================================================================

# Dimensions du monde de jeu (en pixels logiques)
# Ces valeurs définissent l'espace de simulation
fenX = 7000  # Largeur du monde
fenY = 3000  # Hauteur du monde

# Facteur d'échelle pour l'affichage
# - La fenêtre affichée = dimensions / echelle
# - echelle = 5 → fenêtre de 1400x600 pixels
echelle = 5

# Rotation du vaisseau par appui de touche (en degrés)
degres_par_tour = 15

# Vitesses maximales pour un atterrissage réussi
# Si le vaisseau dépasse ces vitesses lors du contact, il crash
max_v_speed = 40  # Vitesse verticale max (pixels/frame)
max_h_speed = 20  # Vitesse horizontale max (pixels/frame)

# =============================================================================
# COULEURS - PALETTE SPATIALE RÉALISTE
# =============================================================================
# Toutes les couleurs sont en format RGB (Rouge, Vert, Bleu)
# Certaines incluent un canal Alpha pour la transparence (RGBA)

# --- Couleurs de base ---
BLANC = (255, 255, 255)
NOIR = (0, 0, 0)
GRIS = (127, 127, 127)
ROUGE = (255, 0, 0)
VERT = (0, 255, 0)
BLEU = (0, 0, 255)
JAUNE = (255, 255, 0)
ORANGE = (255, 127, 0)
BORDEAUX = (255, 100, 100)

# --- Espace (fond étoilé) ---
NOIR_ESPACE = (8, 8, 20)           # Fond du ciel (noir avec teinte bleue)
ETOILE_DIM = (120, 120, 140)       # Étoiles faibles
ETOILE_MEDIUM = (180, 180, 200)    # Étoiles moyennes
ETOILE_BRIGHT = (255, 255, 245)    # Étoiles brillantes
ETOILE_BLEU = (200, 220, 255)      # Étoiles bleues (chaudes)
ETOILE_ROUGE = (255, 200, 180)     # Étoiles rouges (froides)

# --- Mars - Ciel (dégradé du haut vers le bas) ---
MARS_CIEL_HAUT = (15, 8, 25)          # Noir violacé (haut du ciel)
MARS_CIEL_MILIEU = (60, 30, 40)       # Transition violacée
MARS_CIEL_BAS = (140, 80, 60)         # Orange/brun à l'horizon
MARS_ATMOSPHERE = (180, 100, 70, 80)  # Brume atmosphérique (avec alpha)

# --- Mars - Sol (terrain) ---
MARS_SOL_CLAIR = (195, 90, 50)        # Rouge martien clair (crêtes)
MARS_SOL = (160, 70, 40)              # Rouge martien standard
MARS_SOL_SOMBRE = (100, 45, 30)       # Rouge martien sombre (montagnes)
MARS_ROCHE = (70, 35, 25)             # Roches sombres
MARS_OMBRE = (40, 20, 15)             # Ombres profondes

# --- Zone d'atterrissage ---
ZONE_ATTERRISSAGE = (50, 255, 150)    # Vert lumineux (zone sécurisée)
ZONE_ATTERRISSAGE_DIM = (30, 150, 90) # Vert atténué
BALISE_LUMIERE = (255, 255, 200)      # Lumière des balises (jaune clair)

# --- Flammes de propulsion (du centre vers l'extérieur) ---
# Les flammes réalistes sont plus chaudes (bleues) au centre
FLAMME_COEUR = (220, 240, 255)        # Centre bleu-blanc (très chaud ~3000K)
FLAMME_INTERIEUR = (255, 220, 150)    # Jaune-orange (chaud)
FLAMME_MILIEU = (255, 150, 50)        # Orange vif
FLAMME_EXTERIEUR = (255, 80, 30)      # Rouge-orange (plus froid)
FLAMME_BOUT = (200, 50, 20, 150)      # Rouge sombre avec transparence

# --- Effets visuels (explosions et poussière) ---
EXPLOSION_JAUNE = (255, 255, 100)     # Centre de l'explosion
EXPLOSION_ORANGE = (255, 180, 50)     # Milieu de l'explosion
EXPLOSION_ROUGE = (255, 80, 30)       # Bord de l'explosion
POUSSIERE_CLAIRE = (200, 150, 120)    # Poussière martienne claire
POUSSIERE_SOMBRE = (120, 80, 60)      # Poussière martienne sombre

# --- Indicateurs de succès et échec ---
SUCCES_VERT = (50, 255, 120)          # Atterrissage réussi
SUCCES_LUEUR = (100, 255, 150, 100)   # Lueur de succès (avec alpha)
CRASH_ROUGE = (255, 60, 60)           # Crash
CRASH_LUEUR = (255, 100, 100, 100)    # Lueur de crash (avec alpha)

# --- HUD Sci-Fi (Interface utilisateur) ---
HUD_PRIMAIRE = (80, 180, 255)         # Bleu cyan principal
HUD_SECONDAIRE = (60, 140, 200)       # Bleu plus foncé
HUD_ACCENT = (100, 255, 200)          # Cyan-vert accent (valeurs importantes)
HUD_ALERTE = (255, 150, 50)           # Orange alerte (attention)
HUD_DANGER = (255, 80, 80)            # Rouge danger (critique)
HUD_FOND = (15, 25, 40)               # Fond des panneaux (bleu sombre)
HUD_FOND_ALPHA = (15, 25, 40, 200)    # Fond avec transparence
HUD_BORDURE = (60, 120, 180)          # Bordure des panneaux
HUD_TEXTE = (200, 220, 255)           # Texte clair

# --- Trajectoire prédictive ---
TRAJECTOIRE = (100, 200, 255, 150)        # Bleu semi-transparent (normal)
TRAJECTOIRE_DANGER = (255, 100, 100, 150) # Rouge si crash imminent

# =============================================================================
# SCÉNARIOS DE JEU
# =============================================================================
# Chaque scénario définit:
# - surface_mars: Liste de points (x, y) formant le terrain
# - vaisseau: Position et état initial du vaisseau
#
# La zone d'atterrissage est automatiquement détectée comme
# le segment horizontal le plus long du terrain.

# Scénario 0: FACILE - Terrain simple, zone d'atterrissage large
# Idéal pour le début de l'entraînement de l'IA
scenario0 = {
    "surface_mars": [
        (0, fenY - 200),         # Point gauche
        (3000, fenY - 750),      # Début zone d'atterrissage
        (4000, fenY - 750),      # Fin zone d'atterrissage (1000px de large)
        (fenX, fenY - 200)       # Point droit
    ],
    "vaisseau": {
        "x": 2000, "y": fenY - 2500,  # Position: centré au-dessus de la zone
        "h_speed": 0, "v_speed": 0,   # Vitesse: immobile
        "fuel": 1000, "rotate": 0, "power": 0  # Carburant plein, horizontal
    }
}

# Scénario 1: MOYEN - Terrain vallonné
scenario1 = {
    "surface_mars": [
        (0, fenY - 100), (1000, fenY - 500), (1500, fenY - 1500),
        (3000, fenY - 1000), (4000, fenY - 150), (5500, fenY - 150),
        (fenX, fenY - 800)
    ],
    "vaisseau": {
        "x": 2500, "y": fenY - 2700,
        "h_speed": 0, "v_speed": 0,
        "fuel": 550, "rotate": 0, "power": 0
    }
}

# Scénario 2: DIFFICILE - Terrain accidenté, vaisseau en mouvement
scenario2 = {
    "surface_mars": [
        (0, fenY - 100), (1000, fenY - 500), (1500, fenY - 100),
        (3000, fenY - 100), (3500, fenY - 500), (3700, fenY - 200),
        (5000, fenY - 1500), (5800, fenY - 300), (6000, fenY - 1000),
        (fenX, fenY - 2000)
    ],
    "vaisseau": {
        "x": 6500, "y": fenY - 2800,
        "h_speed": -100, "v_speed": 0,    # Se déplace vers la gauche!
        "fuel": 600, "rotate": 90, "power": 0  # Incliné à 90°
    }
}

# Scénario 3: DIFFICILE - Zone d'atterrissage éloignée
scenario3 = {
    "surface_mars": [
        (0, fenY - 100), (1000, fenY - 500), (1500, fenY - 1500),
        (3000, fenY - 1000), (4000, fenY - 150), (5500, fenY - 150),
        (fenX, fenY - 800)
    ],
    "vaisseau": {
        "x": 6500, "y": fenY - 2800,
        "h_speed": -90, "v_speed": 0,
        "fuel": 750, "rotate": 90, "power": 0
    }
}

# Scénario 4: TRÈS DIFFICILE - Terrain très accidenté
scenario4 = {
    "surface_mars": [
        (0, fenY - 1000), (300, fenY - 1500), (350, fenY - 1400),
        (500, fenY - 2000), (800, fenY - 1800), (1000, fenY - 2500),
        (1200, fenY - 2100), (1500, fenY - 2400), (2000, fenY - 1000),
        (2200, fenY - 500), (2500, fenY - 100), (2900, fenY - 800),
        (3000, fenY - 500), (3200, fenY - 1000), (3500, fenY - 2000),
        (3800, fenY - 800), (4000, fenY - 200), (5000, fenY - 200),
        (5500, fenY - 1500), (fenX, fenY - 2800)
    ],
    "vaisseau": {
        "x": 500, "y": fenY - 2700,
        "h_speed": 100, "v_speed": 0,     # Se déplace vers la droite!
        "fuel": 800, "rotate": -90, "power": 0  # Incliné à -90°
    }
}

# Scénario 5: EXPERT - Terrain complexe avec peu de zones plates
scenario5 = {
    "surface_mars": [
        (0, fenY - 1000), (300, fenY - 1500), (350, fenY - 1400),
        (500, fenY - 2100), (1500, fenY - 2100), (2000, fenY - 200),
        (2500, fenY - 500), (2900, fenY - 300), (3000, fenY - 200),
        (3200, fenY - 1000), (3500, fenY - 500), (3800, fenY - 800),
        (4000, fenY - 200), (4200, fenY - 800), (4800, fenY - 600),
        (5000, fenY - 1200), (5500, fenY - 900), (6000, fenY - 500),
        (6500, fenY - 300), (fenX, fenY - 500)
    ],
    "vaisseau": {
        "x": 6500, "y": fenY - 2700,
        "h_speed": -50, "v_speed": 0,
        "fuel": 1000, "rotate": 90, "power": 0
    }
}
