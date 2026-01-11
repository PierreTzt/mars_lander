# Mars Lander IA - Q-Learning V2.0

Un simulateur d'atterrissage sur Mars avec une Intelligence Artificielle par apprentissage par renforcement (Q-Learning).

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Pygame](https://img.shields.io/badge/Pygame-2.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Description

Ce projet simule l'atterrissage d'un vaisseau spatial sur Mars. Une IA basée sur l'algorithme Q-Learning apprend de manière autonome à faire atterrir le vaisseau en toute sécurité sur la zone d'atterrissage.

### Idee originale

**Florent Lannois** - Concept et idee du projet

## Fonctionnalites V2.0 (20 nouvelles features!)

### Gameplay de base
- **Simulation physique realiste** : Gravite, propulsion, inertie
- **Q-Learning avec Experience Replay** : Apprentissage par renforcement optimise
- **6 niveaux de difficulte** : Du facile a l'expert
- **Systeme de replay** : Enregistrement des meilleures performances

### Graphismes avances
- Fond etoile avec scintillement
- Arriere-plan martien avec degrade
- Systeme de particules (flammes, explosions, poussiere)
- Fusee dessinee proceduralement

### 20 Nouvelles Fonctionnalites

| # | Fonctionnalite | Description |
|---|----------------|-------------|
| 1 | **Meteorites dynamiques** | Obstacles qui tombent du ciel, l'IA doit apprendre a les eviter |
| 2 | **Mode nuit + eclairage** | Visibilite reduite avec un spot lumineux sur le vaisseau |
| 3 | **Plusieurs planetes** | Lune, Mars, Terre, Europa, Titan, Venus avec gravites differentes |
| 4 | **Degats progressifs** | Le vaisseau peut subir des dommages partiels avant destruction |
| 5 | **Zones d'atterrissage multiples** | Plusieurs cibles avec recompenses differentes |
| 6 | **Deep Q-Network (DQN)** | Reseau de neurones au lieu de Q-Table classique |
| 7 | **Camera dynamique** | Zoom/dezoom automatique qui suit le vaisseau |
| 8 | **Mode Time Attack** | Chronometre + classement des meilleurs temps |
| 9 | **Tempetes de poussiere** | Reduction de visibilite + perturbations |
| 10 | **Export GIF** | Enregistrer les meilleurs atterrissages en GIF anime |
| 11 | **Mode multijoueur local** | 2 vaisseaux en meme temps, course a l'atterrissage |
| 12 | **Systeme de missions** | Objectifs varies avec recompenses |
| 13 | **Stations de ravitaillement** | Plateformes en vol pour recuperer du carburant |
| 14 | **Algorithme genetique** | Evolution de populations de pilotes |
| 15 | **Terrain destructible** | Les crashes creent des crateres |
| 16 | **Power-ups** | Bonus a collecter : bouclier, fuel, ralenti temporel |
| 17 | **Brouillard de guerre** | Le terrain se revele progressivement |
| 18 | **Mode survie** | Atterrissages successifs avec difficulte croissante, 3 vies |
| 19 | **Editeur de niveaux** | Creer ses propres terrains avec la souris |
| 20 | **Dashboard statistiques** | Graphiques detailles de l'apprentissage |

## Installation

### Prerequis

- Python 3.8 ou superieur
- pip (gestionnaire de paquets Python)

### Installation des dependances

```bash
pip install pygame numpy
# Optionnel pour l'export GIF:
pip install Pillow
```

### Lancement

```bash
# Version classique
python mars_lander.py

# Version 2.0 avec toutes les fonctionnalites
python mars_lander_v2.py
```

## Controles

### Controles de base

| Touche | Action |
|--------|--------|
| **ESPACE** | Redemarrer la simulation |
| **Fleches gauche/droite** | Rotation du vaisseau (mode manuel) |
| **1-5** | Puissance des moteurs (mode manuel) |
| **P** | Pause/Reprise |
| **+/-** | Ajuster la vitesse de simulation |
| **F1-F6** | Changer de scenario (difficulte) |

### Nouvelles fonctionnalites (V2.0)

| Touche | Action |
|--------|--------|
| **S** | Sons on/off |
| **V** | Vent on/off |
| **T** | Trajectoire predictive on/off |
| **N** | Mode nuit on/off |
| **M** | Meteorites on/off |
| **B** | Brouillard de guerre on/off |
| **C** | Camera dynamique on/off |
| **O** | Declencher une tempete |
| **U** | Power-ups on/off |
| **G** | Demarrer/Arreter enregistrement GIF |
| **E** | Ouvrir l'editeur de niveaux |
| **D** | Afficher le dashboard statistiques |
| **F7** | Mode Time Attack on/off |
| **F8** | Mode Survie on/off |
| **F9** | Missions on/off |
| **1-6** | Changer de planete |

## Configuration

Le fichier `data.py` permet de configurer toutes les fonctionnalites :

### Mode de jeu
```python
ia_active = True  # True = IA, False = Manuel
```

### Hyperparametres Q-Learning
```python
alpha = 0.1          # Taux d'apprentissage
gamma = 0.9          # Facteur de discount
epsilon = 0.8        # Taux d'exploration initial
epsilon_decay = 0.99991  # Decroissance de l'exploration
```

### Nouvelles fonctionnalites
```python
# Meteorites
meteorites_actif = False
meteorites_spawn_rate = 0.02

# Mode nuit
mode_nuit_actif = False

# Planete (moon, mars, earth, europa, titan, venus)
planete_actuelle = 'mars'

# Deep Q-Network
dqn_actif = False

# Et bien plus...
```

## Architecture du projet

```
mars_lander-main/
├── mars_lander.py      # Version classique
├── mars_lander_v2.py   # Version 2.0 avec 20 features
├── data.py             # Configuration et constantes
├── vaisseau.py         # Physique du vaisseau
├── surface.py          # Terrain et zone d'atterrissage
├── jeu.py              # Logique du jeu
├── ia_learning.py      # Algorithme Q-Learning
├── affichage.py        # Rendu graphique
├── game_systems.py     # Nouveaux systemes de jeu
├── advanced_ai.py      # DQN et algorithme genetique
├── extra_features.py   # Editeur, dashboard, GIF export
├── historique/         # Sauvegardes Q-Table
└── README.md
```

## Algorithme Q-Learning

### Equation de Bellman
```
Q(s,a) = Q(s,a) + alpha x (r + gamma x max(Q(s',a')) - Q(s,a))
```

- **s** : Etat actuel
- **a** : Action effectuee
- **r** : Recompense recue
- **s'** : Nouvel etat
- **alpha** : Taux d'apprentissage
- **gamma** : Facteur de discount

### Deep Q-Network (DQN)

La version 2.0 inclut un agent DQN qui utilise un reseau de neurones pour approximer la fonction Q. Avantages :
- Meilleure generalisation
- Gestion d'espaces d'etats continus
- Experience Replay pour stabiliser l'apprentissage

### Algorithme Genetique

Alternative au Q-Learning :
- Population de "pilotes" avec des comportements differents
- Selection naturelle des meilleurs
- Croisement et mutation pour explorer de nouvelles strategies

## Planetes disponibles

| Planete | Gravite (m/s2) | Description |
|---------|---------------|-------------|
| Lune | 1.62 | Tres faible gravite, pas d'atmosphere |
| Mars | 3.72 | Gravite faible, atmosphere tenue |
| Terre | 9.81 | Gravite standard, atmosphere dense |
| Europa | 1.31 | Surface glacee, tres faible gravite |
| Titan | 1.35 | Atmosphere tres dense, vents forts |
| Venus | 8.87 | Gravite forte, pression ecrasante |

## Conditions d'atterrissage

Pour un atterrissage reussi :
- Etre dans la **zone d'atterrissage** (segment horizontal)
- **Angle = 0 degres** (vaisseau horizontal)
- **Vitesse verticale inferieure ou egale a 40** pixels/frame
- **Vitesse horizontale inferieure ou egale a 20** pixels/frame

## Licence

Ce projet est sous licence MIT.

## Credits

- **Idee originale** : Florent Lannois
- **Developpement et ameliorations** : Projet collaboratif avec Claude AI
- **Inspire par** : Le challenge CodinGame "Mars Lander"

---

*Fait avec Python et Pygame - Version 2.0 avec 20 nouvelles fonctionnalites!*
