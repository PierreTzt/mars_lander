"""
Les différentes versions jouables de Mars Lander IA.

Lancement depuis la racine du dépôt : `python lancer.py <version>`
ou `python -m versions.<version>`.
"""

import sys

# Les versions affichent des caractères Unicode dans la console : sous Windows,
# une sortie redirigée (fichier, pipe) plante sans cette précaution.
for flux in (sys.stdout, sys.stderr):
    if hasattr(flux, "reconfigure"):
        flux.reconfigure(errors="replace")
