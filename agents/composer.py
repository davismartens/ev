import json
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
import re

BASE_DIR = Path("agents")

class Composer:


    @staticmethod
    def _load_template(file_name: str, base_dir: str = "agents", **template_vars) -> str:
        base_dir = Path(base_dir)
        if not base_dir.is_dir():
            raise FileNotFoundError(f"Template directory not found: {base_dir}")

        md_path = base_dir / f"{file_name}.md"
        j2_path = base_dir / f"{file_name}.j2"

        if md_path.is_file():
            return md_path.read_text(encoding="utf-8").strip()

        if j2_path.is_file():
            env = Environment(
                loader=FileSystemLoader(str(base_dir)),
                autoescape=False,
                trim_blocks=True,
                lstrip_blocks=True,
            )
            template = env.get_template(f"{file_name}.j2")
            return template.render(**template_vars).strip()

        raise FileNotFoundError(
            f"No template found for {file_name}.md or {file_name}.j2 in {base_dir}"
        )


 