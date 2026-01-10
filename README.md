# Mars Lander IA - Q-Learning

Un simulateur d'atterrissage sur Mars avec une Intelligence Artificielle par apprentissage par renforcement (Q-Learning).

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Pygame](https://img.shields.io/badge/Pygame-2.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Description

Ce projet simule l'atterrissage d'un vaisseau spatial sur Mars. Une IA basée sur l'algorithme Q-Learning apprend de manière autonome à faire atterrir le vaisseau en toute sécurité sur la zone d'atterrissage.

### Idée originale

**Florent Lannois** - Concept et idée du projet

### Fonctionnalités

- **Simulation physique réaliste** : Gravité, propulsion, inertie
- **Q-Learning avec Experience Replay** : Apprentissage par renforcement optimisé
- **Graphismes avancés** :
  - Fond étoilé avec scintillement
  - Arrière-plan martien avec dégradé
  - Système de particules (flammes, explosions, poussière)
  - Fusée dessinée procéduralement
- **Effets météo** : Vent aléatoire qui perturbe le vaisseau
- **Effets sonores** : Sons procéduraux pour propulsion, crash et succès
- **Trajectoire prédictive** : Visualisation de la trajectoire future
- **Graphique d'apprentissage** : Évolution des récompenses en temps réel
- **6 niveaux de difficulté** : Du facile à l'expert
- **Système de replay** : Enregistrement des meilleures performances

## Installation

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### Installation des dépendances

```bash
pip install pygame numpy
```

### Lancement

```bash
python mars_lander.py
```

## Contrôles

| Touche | Action |
|--------|--------|
| **ESPACE** | Redémarrer la simulation |
| **Flèches ←/→** | Rotation du vaisseau (mode manuel) |
| **1-5** | Puissance des moteurs (mode manuel) |
| **P** | Pause/Reprise |
| **+/-** | Ajuster la vitesse de simulation |
| **F1-F6** | Changer de scénario (difficulté) |
| **S** | Sons on/off |
| **V** | Vent on/off |
| **T** | Trajectoire prédictive on/off |

## Configuration

Le fichier `data.py` permet de configurer :

### Mode de jeu
```python
ia_active = True  # True = IA, False = Manuel
```

### Hyperparamètres Q-Learning
```python
alpha = 0.1          # Taux d'apprentissage
gamma = 0.9          # Facteur de discount
epsilon = 0.8        # Taux d'exploration initial
epsilon_decay = 0.99991  # Décroissance de l'exploration
```

### Fonctionnalités
```python
vent_actif = True           # Vent aléatoire
trajectoire_active = True   # Trajectoire prédictive
sons_actifs = True          # Effets sonores
graphique_actif = True      # Graphique d'apprentissage
```

## Architecture du projet

```
mars_lander-main/
├── mars_lander.py    # Point d'entrée, boucle principale
├── data.py           # Configuration et constantes
├── vaisseau.py       # Physique du vaisseau
├── surface.py        # Terrain et zone d'atterrissage
├── jeu.py            # Logique du jeu
├── ia_learning.py    # Algorithme Q-Learning
├── affichage.py      # Rendu graphique
├── historique/       # Sauvegardes Q-Table
└── README.md
```

## Algorithme Q-Learning

### Équation de Bellman
```
Q(s,a) = Q(s,a) + α × (r + γ × max(Q(s',a')) - Q(s,a))
```

- **s** : État actuel
- **a** : Action effectuée
- **r** : Récompense reçue
- **s'** : Nouvel état
- **α** : Taux d'apprentissage
- **γ** : Facteur de discount

### Système de récompenses

| Événement | Récompense |
|-----------|------------|
| Atterrissage réussi | +100 |
| Crash | -50 |
| Se rapprocher de la zone | +2 |
| S'éloigner de la zone | -1 |
| Être dans la zone | +5 |
| Vitesse contrôlée | +3 |
| Angle correct | +2 |
| Bonus carburant | +0.05 × fuel |

## Conditions d'atterrissage

Pour un atterrissage réussi :
- Être dans la **zone d'atterrissage** (segment horizontal)
- **Angle = 0°** (vaisseau horizontal)
- **Vitesse verticale ≤ 40** pixels/frame
- **Vitesse horizontale ≤ 20** pixels/frame

## Captures d'écran

Le jeu affiche :
- Un HUD sci-fi avec toutes les informations
- Un mini-radar montrant la position relative
- Un graphique des récompenses en temps réel
- Un indicateur de vent
- Une trajectoire prédictive en pointillés

## Licence

Ce projet est sous licence MIT.

## Crédits

- **Idée originale** : Florent Lannois
- **Développement et améliorations** : Projet collaboratif avec Claude AI
- **Inspiré par** : Le challenge CodinGame "Mars Lander"

---

*Fait avec Python et Pygame*
