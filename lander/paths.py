"""
Chemins du projet.

Tout ce que le jeu écrit (Q-tables, modèle PyTorch, succès, GIF...) va dans
le dossier `saves/` à la racine du dépôt, quel que soit le dossier courant.
"""

from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
DOSSIER_SAUVEGARDES = RACINE / "saves"


def chemin_sauvegarde(nom: str) -> str:
    """Retourne le chemin d'un fichier dans saves/, en créant le dossier si besoin."""
    DOSSIER_SAUVEGARDES.mkdir(exist_ok=True)
    return str(DOSSIER_SAUVEGARDES / nom)
