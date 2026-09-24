"""
Lanceur de Mars Lander IA.

Usage :
    python lancer.py            # version PRO (recommandée)
    python lancer.py classique  # ou v2, ultimate, deluxe, pro
"""

import importlib
import sys

VERSIONS = {
    "pro": "PyTorch DQN + graphiques temps réel (recommandée)",
    "classique": "Q-Learning avec table, version de base",
    "v2": "20 fonctionnalités : planètes, météorites, algorithme génétique...",
    "ultimate": "Effets visuels avancés, progression de l'IA visible",
    "deluxe": "Menu, succès, cartes de chaleur",
}


def main() -> None:
    version = sys.argv[1].lower() if len(sys.argv) > 1 else "pro"
    if version not in VERSIONS:
        print(f"Version inconnue : {version}\n\nVersions disponibles :")
        for nom, description in VERSIONS.items():
            print(f"  {nom:<10} {description}")
        sys.exit(1)

    module = importlib.import_module(f"versions.{version}")
    module.main()


if __name__ == "__main__":
    main()
