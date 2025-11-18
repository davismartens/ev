import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ev.utils.logger import logger

# Root directory that contains the EVALS folder
ROOT_DIR = Path(__file__).resolve().parent.parent
EVALS_ROOT = ROOT_DIR / "EVALS"


def snapshot_prompts(test_dir: Path) -> str:
    versions_dir = test_dir / "versions"
    versions_dir.mkdir(exist_ok=True)

    system_src = test_dir / "system_prompt.j2"
    user_src = test_dir / "user_prompt.j2"

    missing = [str(p.name) for p in [system_src, user_src] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Cannot snapshot prompts, missing: {', '.join(missing)} "
            f"in {test_dir}"
        )

    u = str(uuid.uuid4())
    h = u[:8]
    timestamp = datetime.now().strftime("%d %b %Y %H-%M-%S")
    timestamp_safe = timestamp.replace(":", "-")

    # Prefix version id with "base - ..." so it is clearly the initial snapshot
    version_id = f"base - {timestamp_safe}"
    version_dir = versions_dir / version_id
    version_dir.mkdir(parents=True, exist_ok=False)

    system_dst = version_dir / "system_prompt.j2"
    user_dst = version_dir / "user_prompt.j2"

    system_dst.write_text(system_src.read_text(encoding="utf-8"), encoding="utf-8")
    user_dst.write_text(user_src.read_text(encoding="utf-8"), encoding="utf-8")

    logger.info(
        f"[INIT] Creating initial base snapshot version {version_id} in {version_dir}"
    )
    return version_id



def create_version_from_prompts(
    test_dir: Path,
    system_src: str,
    user_src: str,
    pass_rate: float,
) -> str:
    versions_dir = test_dir / "versions"
    versions_dir.mkdir(exist_ok=True)
    log_path = versions_dir / "log.json"

    u = str(uuid.uuid4())
    h = u[:8]
    timestamp = datetime.now().strftime("%d %b %Y %H-%M-%S")
    timestamp_safe = timestamp.replace(":", "-")
    version_id = f"{h} - {timestamp_safe}"
    version_dir = versions_dir / version_id
    version_dir.mkdir(parents=True, exist_ok=False)

    (version_dir / "system_prompt.j2").write_text(system_src, encoding="utf-8")
    (version_dir / "user_prompt.j2").write_text(user_src, encoding="utf-8")

    entries: List[Dict[str, Any]] = []
    if log_path.exists():
        entries = json.loads(log_path.read_text(encoding="utf-8"))

    for entry in entries:
        entry["is_active"] = False

    entries.append(
        {
            "version": version_id,
            "pass_rate": pass_rate,
            "is_active": True,
        }
    )

    log_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return version_id


def load_active_version(test_dir: Path) -> str:
    versions_dir = test_dir / "versions"
    versions_dir.mkdir(exist_ok=True)

    log_path = versions_dir / "log.json"

    # First time - no log.json yet
    if not log_path.exists():
        candidates = [p for p in versions_dir.iterdir() if p.is_dir()]

        if candidates:
            latest = max(candidates, key=lambda p: p.stat().st_mtime)
            version_id = latest.name
            logger.info(
                f"[INIT] Found existing version directories without log.json - "
                f"using latest as active base: {version_id}"
            )
        else:
            # No versions at all - create initial base snapshot
            version_id = snapshot_prompts(test_dir)

        entries: List[Dict[str, Any]] = [
            {
                "version": version_id,
                "pass_rate": 0.0,
                "is_active": True,
            }
        ]
        log_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        return version_id

    # Normal path - log.json exists
    entries = json.loads(log_path.read_text(encoding="utf-8"))

    # Try to find an active version
    for entry in entries:
        if entry.get("is_active"):
            return entry["version"]

    # No active version marked; choose or create one
    if entries:
        entries[0]["is_active"] = True
        version_id = entries[0]["version"]
    else:
        candidates = [p for p in versions_dir.iterdir() if p.is_dir()]
        if candidates:
            latest = max(candidates, key=lambda p: p.stat().st_mtime)
            version_id = latest.name
            logger.info(
                f"[INIT] No active version in log.json - "
                f"using latest directory as active: {version_id}"
            )
        else:
            version_id = snapshot_prompts(test_dir)

        entries = [
            {
                "version": version_id,
                "pass_rate": 0.0,
                "is_active": True,
            }
        ]

    log_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return version_id

