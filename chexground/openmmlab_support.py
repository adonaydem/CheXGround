import importlib
import sys
from pathlib import Path


def _purge_openmmlab_modules() -> None:
    for module_name in list(sys.modules):
        if (
            module_name == "mmcv"
            or module_name.startswith("mmcv.")
            or module_name == "mmdet"
            or module_name.startswith("mmdet.")
        ):
            sys.modules.pop(module_name, None)


def _purge_shadowed_datasets_module(expected_mmdet_prefix: str) -> None:
    datasets_module = sys.modules.get("datasets")
    datasets_file = _module_file(datasets_module)
    shadow_prefix = f"{expected_mmdet_prefix}/datasets"
    if datasets_file and _path_is_under(datasets_file, shadow_prefix):
        for module_name in list(sys.modules):
            if module_name == "datasets" or module_name.startswith("datasets."):
                sys.modules.pop(module_name, None)


def _module_file(module) -> str:
    module_file = getattr(module, "__file__", "") or ""
    if not module_file:
        return ""
    return str(Path(module_file).resolve())


def _path_is_under(module_file: str, expected_prefix: str) -> bool:
    return module_file == expected_prefix or module_file.startswith(expected_prefix + "/")


def _loaded_openmmlab_modules_match(expected_mmcv_prefix: str, expected_mmdet_prefix: str) -> bool:
    for module_name, module in list(sys.modules.items()):
        if module is None:
            continue
        if module_name == "mmcv" or module_name.startswith("mmcv."):
            expected_prefix = expected_mmcv_prefix
        elif module_name == "mmdet" or module_name.startswith("mmdet."):
            expected_prefix = expected_mmdet_prefix
        else:
            continue
        module_file = _module_file(module)
        if not module_file or not _path_is_under(module_file, expected_prefix):
            return False
    return True


def ensure_openmmlab_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    mmcv_root = repo_root / "mmcv"
    mmdet_root = repo_root / "mmdet"

    mmcv_root_str = str(mmcv_root)
    mmdet_root_str = str(mmdet_root)
    repo_root_str = str(repo_root)
    expected_mmcv_prefix = str((mmcv_root / "mmcv").resolve())
    expected_mmdet_prefix = str((repo_root / "mmdet").resolve())

    if mmcv_root_str in sys.path:
        sys.path.remove(mmcv_root_str)
    sys.path.insert(0, mmcv_root_str)

    if mmdet_root_str in sys.path:
        sys.path.remove(mmdet_root_str)

    if repo_root_str not in sys.path:
        sys.path.insert(1, repo_root_str)

    if not _loaded_openmmlab_modules_match(expected_mmcv_prefix, expected_mmdet_prefix):
        _purge_openmmlab_modules()
    _purge_shadowed_datasets_module(expected_mmdet_prefix)

    mmcv = importlib.import_module("mmcv")
    mmcv_file = _module_file(mmcv)
    mmcv_version = getattr(mmcv, "__version__", None)
    if mmcv_version is None or not _path_is_under(mmcv_file, expected_mmcv_prefix):
        raise ImportError(
            f"Resolved mmcv without __version__: file={mmcv_file or '<namespace>'}. "
            f"Expected vendored package under {expected_mmcv_prefix}."
        )

    mmdet = importlib.import_module("mmdet")
    mmdet_file = _module_file(mmdet)
    if not _path_is_under(mmdet_file, expected_mmdet_prefix):
        raise ImportError(
            f"Resolved mmdet from unexpected path: file={mmdet_file or '<namespace>'}. "
            f"Expected vendored package under {expected_mmdet_prefix}."
        )

    return repo_root


OPENMMLAB_ROOT = ensure_openmmlab_paths()

__all__ = ["OPENMMLAB_ROOT", "ensure_openmmlab_paths"]
