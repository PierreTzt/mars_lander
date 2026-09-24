"""Importe tous les modules de lander/ et versions/ pour détecter les erreurs d'import."""

import importlib
import pkgutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lander  # noqa: E402
import versions  # noqa: E402

for paquet in (lander, versions):
    for module in pkgutil.iter_modules(paquet.__path__):
        importlib.import_module(f"{paquet.__name__}.{module.name}")
        print(f"ok  {paquet.__name__}.{module.name}")
