"""InsureRAG - RAG sur des conditions generales d'assurance habitation."""

from __future__ import annotations

import sys
from pathlib import Path

# config.py est a la racine du projet, pas dans le package.
# du coup on ajoute la racine au path pour pouvoir faire "import config"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

__version__ = "0.1.0"
