# Mars Lander IA - Q-Learning DELUXE EDITION

Un simulateur d'atterrissage sur Mars avec une Intelligence Artificielle par apprentissage par renforcement (Q-Learning).

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Pygame](https://img.shields.io/badge/Pygame-2.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Description

Ce projet simule l'atterrissage d'un vaisseau spatial sur Mars. Une IA basee sur l'algorithme Q-Learning apprend de maniere autonome a faire atterrir le vaisseau en toute securite sur la zone d'atterrissage.

### Idee originale

**Florent Lannois** - Concept et idee du projet

## Versions disponibles

| Version | Fichier | Description |
|---------|---------|-------------|
| **Classique** | `mars_lander.py` | Version de base avec Q-Learning |
| **V2.0** | `mars_lander_v2.py` | 20 nouvelles fonctionnalites |
| **Ultimate** | `mars_lander_ultimate.py` | Effets visuels avances, progression IA visible |
| **Deluxe** | `mars_lander_deluxe.py` | Experience complete avec menu, achievements, heatmaps |

## Edition Deluxe - Experience Complete

L'edition Deluxe offre l'experience la plus complete :

### Menu Principal
- Interface graphique intuitive
- Boutons pour choisir le mode de jeu
- Animation d'etoiles en arriere-plan

### Systeme d'Achievements (17 succes)

| Achievement | Description |
|-------------|-------------|
| Premier Atterrissage | Reussir son premier atterrissage |
| 10 Atterrissages | Cumuler 10 atterrissages reussis |
| 100 Atterrissages | Cumuler 100 atterrissages reussis |
| Serie de 5 | 5 atterrissages consecutifs |
| Serie de 10 | 10 atterrissages consecutifs |
| Serie de 25 | 25 atterrissages consecutifs |
| Atterrissage Parfait | Vitesse < 10 et angle = 0 |
| Maitre du Vent | Atterrir avec le vent actif |
| Explorateur Nocturne | Atterrir en mode nuit |
| Esquive de Meteorite | Eviter 10 meteorites |
| Survivant | Survivre 5 vagues en mode survie |
| Econome | Atterrir avec > 50% de fuel |
| Temoin de l'Aube | Jouer pendant 1 heure |
| Explorateur | Visiter toutes les planetes |
| Collectionneur | Ramasser 50 power-ups |
| Precision Chirurgicale | 10 atterrissages sans crash |
| Legendaire | Debloquer tous les achievements |

### Heatmaps de Visualisation
- Carte de chaleur des crashes (rouge)
- Carte de chaleur des atterrissages (vert)
- Visualisation de la progression de l'apprentissage

### Systeme de Particules
- Flammes de propulsion realistes
- Explosions spectaculaires
- Effets de celebration lors des succes
- Poussiere a l'atterrissage

### Statistiques en Temps Reel
- Nombre total d'episodes
- Taux de reussite
- Meilleure serie consecutive
- Temps de jeu

## 20 Nouvelles Fonctionnalites (V2.0+)

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

# Version Ultimate (effets visuels avances)
python mars_lander_ultimate.py

# Version Deluxe (RECOMMANDEE - experience complete)
python mars_lander_deluxe.py
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
| **ESC** | Retour au menu (Deluxe) |

### Fonctionnalites avancees

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
| **H** | Afficher/Masquer heatmap (Deluxe) |
| **A** | Afficher achievements (Deluxe) |
| **F7** | Mode Time Attack on/off |
| **F8** | Mode Survie on/off |
| **F9** | Missions on/off |

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
├── mars_lander.py          # Version classique
├── mars_lander_v2.py       # Version 2.0 avec 20 features
├── mars_lander_ultimate.py # Version avec effets visuels avances
├── mars_lander_deluxe.py   # Version complete (RECOMMANDEE)
├── data.py                 # Configuration et constantes
├── vaisseau.py             # Physique du vaisseau
├── surface.py              # Terrain et zone d'atterrissage
├── jeu.py                  # Logique du jeu
├── ia_learning.py          # Algorithme Q-Learning
├── affichage.py            # Rendu graphique
├── game_systems.py         # Nouveaux systemes de jeu
├── advanced_ai.py          # DQN et algorithme genetique
├── extra_features.py       # Editeur, dashboard, GIF export
├── historique/             # Sauvegardes Q-Table
├── achievements.json       # Sauvegarde des succes (auto-genere)
└── README.md
```

## Algorithme Q-Learning

### Equation de Bellman
```
Q(s,a) = Q(s,a) + alpha x (r + gamma x max(Q(s',a')) - Q(s,a))
```

- **s** : Etat actuel (position, vitesse, angle, fuel)
- **a** : Action effectuee (angle, puissance)
- **r** : Recompense recue
- **s'** : Nouvel etat
- **alpha** : Taux d'apprentissage (0.1)
- **gamma** : Facteur de discount (0.9)

### Experience Replay

L'IA stocke ses experiences passees et rejoue des mini-batches aleatoires pour :
- Briser la correlation entre experiences consecutives
- Stabiliser l'apprentissage
- Reutiliser les experiences rares

### Deep Q-Network (DQN)

La version avancee utilise un reseau de neurones pour approximer la fonction Q :
- Couche d'entree : 8 neurones (etat)
- Couches cachees : 64 -> 64 neurones (ReLU)
- Couche de sortie : actions possibles

### Algorithme Genetique

Alternative au Q-Learning basee sur l'evolution :
- Population de 50 "pilotes" avec des poids aleatoires
- Evaluation fitness basee sur la distance a la cible
- Selection des meilleurs (top 20%)
- Croisement et mutation pour la diversite

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
- **Vitesse verticale <= 40** pixels/frame
- **Vitesse horizontale <= 20** pixels/frame

## Progression de l'IA

L'IA commence avec un comportement aleatoire (epsilon = 0.8 = 80% aleatoire).
Au fil des episodes, elle apprend et reduit son exploration :
- Episode 1-100 : Exploration intense, nombreux crashes
- Episode 100-500 : Apprentissage des bases
- Episode 500-1000 : Amelioration du taux de reussite
- Episode 1000+ : Maitrise et optimisation

Visualisez la progression avec :
- Les heatmaps (H) montrant l'evolution des zones de crash/atterrissage
- Les statistiques en temps reel (taux de reussite, epsilon)
- Le dashboard statistiques (D) pour les graphiques detailles

## Licence

Ce projet est sous licence MIT.

## Credits

- **Idee originale** : Florent Lannois
- **Developpement** : Pierre Touzet
- **Assistance IA** : Claude (Anthropic)
- **Inspire par** : Le challenge CodinGame "Mars Lander"

---

*Fait avec Python et Pygame - DELUXE EDITION avec menu, achievements, heatmaps et effets visuels!*
