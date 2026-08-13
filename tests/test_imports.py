"""Verify package modules import successfully."""

import importlib
import pkgutil

import multimodal_agent


def _discover_submodules(package_name: str) -> list[str]:
    """Return import paths for all submodules under a package."""
    package = importlib.import_module(package_name)
    modules = [package_name]
    if not hasattr(package, "__path__"):
        return modules

    for module_info in pkgutil.walk_packages(package.__path__, prefix=f"{package_name}."):
        modules.append(module_info.name)
    return modules


def test_all_modules_import() -> None:
    """Every package module should import without error."""
    failures: list[str] = []
    for module_name in _discover_submodules("multimodal_agent"):
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{module_name}: {exc}")

    assert not failures, "Import failures:\n" + "\n".join(failures)
