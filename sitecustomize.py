"""Windows DLL path bootstrap for local PyTorch installs."""

from __future__ import annotations

import os
import site
import sys
from pathlib import Path


def _torch_lib_dirs() -> list[Path]:
    candidates: list[str] = []
    candidates.extend(path for path in sys.path if path)

    try:
        candidates.extend(site.getsitepackages())
    except AttributeError:
        pass

    user_site = site.getusersitepackages()
    if user_site:
        candidates.append(user_site)

    lib_dirs: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        torch_lib = Path(candidate) / "torch" / "lib"
        if not torch_lib.is_dir():
            continue
        resolved = str(torch_lib.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        lib_dirs.append(torch_lib)
    return lib_dirs


if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    for dll_dir in _torch_lib_dirs():
        os.add_dll_directory(str(dll_dir))
