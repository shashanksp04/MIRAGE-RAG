from __future__ import annotations

import sys
from pathlib import Path


def add_project_root_to_syspath(rag_agent_dir: Path) -> None:
    """
    rag_agent_dir = .../parent/rag_agent
    We need to add .../parent to sys.path so `import rag_agent...` works.
    """
    rag_agent_dir = Path(rag_agent_dir).resolve()
    project_root = rag_agent_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))