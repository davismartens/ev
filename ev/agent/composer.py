from pathlib import Path
from jinja2 import Environment, FileSystemLoader

BASE_DIR = Path(__file__).resolve().parents[1]

class Composer:
    @staticmethod
    def _load_template(file_name: str, sub_dir: str, **template_vars) -> str:
        base_path = BASE_DIR / sub_dir
        if not base_path.is_dir():
            raise FileNotFoundError(f"Template directory not found: {base_path}")

        md_path = base_path / f"{file_name}.md"
        j2_path = base_path / f"{file_name}.j2"

        if md_path.is_file():
            return md_path.read_text(encoding="utf-8").strip()

        if j2_path.is_file():
            env = Environment(
                loader=FileSystemLoader(str(base_path)),
                autoescape=False,
                trim_blocks=True,
                lstrip_blocks=True,
            )
            template = env.get_template(f"{file_name}.j2")
            return template.render(**template_vars).strip()

        raise FileNotFoundError(
            f"No template found for {file_name}.md or {file_name}.j2 in {base_path}"
        )
