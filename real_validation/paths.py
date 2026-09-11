"""Result locations for a source checkout and a portable installation."""
import os
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent


def default_results_root(app_dir=None):
    app = Path(app_dir or APP_DIR).resolve()
    root = app.parent
    override = os.environ.get('REAL_VALIDATION_RESULTS')
    if override:
        return resolve_results_root(override, app)
    if (app/'PACKAGE_MANIFEST.json').is_file():
        return root/'results'
    if (root/'src/registry/paths.py').is_file():
        from src.registry.paths import ProjectPaths
        return ProjectPaths.load(repo_root=root).runs_root/'validation'
    return root/'results'


def resolve_results_root(value, app_dir=None):
    if not str(value).strip():
        raise ValueError('请填写实验结果保存根目录')
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(app_dir or APP_DIR).resolve().parent/path
    return path.resolve()
