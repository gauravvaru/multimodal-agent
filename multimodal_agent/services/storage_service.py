from __future__ import annotations
import tempfile
from pathlib import Path
from multimodal_agent.utilities.security import sanitize_filename

_upload_storage: TempStorageService | None = None

def get_upload_storage() -> TempStorageService:
    global _upload_storage
    if _upload_storage is None:
        _upload_storage = TempStorageService()
    return _upload_storage

def reset_upload_storage_for_tests() -> None:
    global _upload_storage
    _upload_storage = None

def load_artifact_bytes(source: str) -> bytes:
    stripped = source.strip()
    if not stripped:
        raise ValueError("Artifact source is required")

    if stripped.startswith("upload://"):
        return get_upload_storage().read_bytes(stripped)

    path = Path(stripped)
    if path.is_file():
        return path.read_bytes()

    raise FileNotFoundError(f"Artifact not found for source '{stripped}'")


class StorageService:
    async def store(self, filename: str, content: bytes) -> str:
        raise NotImplementedError
    async def read(self, reference: str) -> bytes:
        raise NotImplementedError

class TempStorageService(StorageService):
    def __init__(self, *, base_dir: str | None = None) -> None:
        self._base_dir = Path(base_dir or tempfile.mkdtemp(prefix="multimodal-agent-"))
        self._base_dir.mkdir(parents=True, exist_ok=True)

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def store_bytes(self, filename: str, content: bytes) -> str:
        safe_name = sanitize_filename(filename)
        target = self._resolve_safe_path(safe_name)
        target.write_bytes(content)
        return f"upload://{safe_name}"

    def read_bytes(self, reference: str) -> bytes:
        filename = reference.removeprefix("upload://")
        target = self._resolve_safe_path(sanitize_filename(filename))
        return target.read_bytes()

    async def store(self, filename: str, content: bytes) -> str:
        return self.store_bytes(filename, content)

    async def read(self, reference: str) -> bytes:
        return self.read_bytes(reference)

    def _resolve_safe_path(self, filename: str) -> Path:
        safe_name = sanitize_filename(filename)
        target = (self._base_dir / safe_name).resolve()
        base = self._base_dir.resolve()
        target.relative_to(base)
        return target
