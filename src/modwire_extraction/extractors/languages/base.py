import abc
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict

from ...dependency.resolution import resolve_imports
from ...identity import (
    DuplicateIdentityError,
    FileId,
    ModuleId,
    file_id_for_path,
    module_id_for_path,
)
from ..source import SourceFile


class SourceExtraction(BaseModel):
    model_config = ConfigDict(frozen=True)

    files: dict[FileId, SourceFile]
    modules: dict[ModuleId, FileId]
    files_found: int
    files_excluded: int = -1

    def files_dict(self) -> dict[FileId, SourceFile]:
        return dict(self.files)


@dataclass(frozen=True)
class BatchConfig:
    size: int = 500
    parallel_threshold: int = 0
    parallel_size: int = 0
    max_workers: int = 1
    output_format: Literal["json", "jsonl"] = "json"


@dataclass(frozen=True)
class ExtractorRuntime:
    language: str
    file_extensions: tuple[str, ...]
    command: tuple[str, ...]
    script_path: Path


class SourceExtractor(abc.ABC):
    excluded_dir_names = frozenset(
        {
            ".git",
            ".hg",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            ".svn",
            ".venv",
            "__pycache__",
            "build",
            "coverage",
            "dist",
            "ignored",
            "node_modules",
            "vendor",
        }
    )

    @property
    @abc.abstractmethod
    def runtime(self) -> ExtractorRuntime:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def batch_config(self) -> BatchConfig:
        raise NotImplementedError

    def has_source_files(self, root: Path) -> bool:
        resolved_root = root.resolve()
        if not resolved_root.is_dir():
            raise ValueError(f"Source root is not a directory: {root}")

        source_paths, _ = self._discover_source_files(resolved_root)
        return bool(source_paths)

    def extract_source(
        self,
        root: Path,
        *,
        count_excluded_files: bool = False,
    ) -> SourceExtraction:
        resolved_root = root.resolve()
        if not resolved_root.is_dir():
            raise ValueError(f"Source root is not a directory: {root}")

        source_paths, files_excluded = self._discover_source_files(
            resolved_root,
            count_excluded_files=count_excluded_files,
        )
        files: dict[FileId, SourceFile] = {}
        batches = self._source_batches(source_paths)

        if self._uses_parallel_batches(len(source_paths)):
            max_workers = max(1, self.batch_config.max_workers)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for extracted in executor.map(
                    self._extract_batch,
                    repeat(resolved_root),
                    batches,
                ):
                    self._merge_files(files, extracted)
        else:
            for batch_paths in batches:
                self._merge_files(
                    files,
                    self._extract_batch(resolved_root, batch_paths),
                )

        modules: dict[ModuleId, FileId] = {}
        for file_id, source_file in files.items():
            existing = modules.get(source_file.module_id)
            if existing is not None:
                raise DuplicateIdentityError(
                    "module",
                    source_file.module_id,
                    existing,
                    file_id,
                )
            modules[source_file.module_id] = file_id

        files = resolve_imports(files)

        return SourceExtraction(
            files=files,
            modules=modules,
            files_found=len(source_paths),
            files_excluded=files_excluded,
        )

    def _source_batches(self, source_paths: list[Path]) -> list[list[Path]]:
        batch_size = self._batch_size(len(source_paths))
        return [
            source_paths[start : start + batch_size]
            for start in range(0, len(source_paths), batch_size)
        ]

    def _batch_size(self, source_count: int) -> int:
        if self._uses_parallel_batches(source_count) and self.batch_config.parallel_size:
            return max(1, self.batch_config.parallel_size)
        return max(1, self.batch_config.size)

    def _uses_parallel_batches(self, source_count: int) -> bool:
        return (
            self.batch_config.parallel_threshold > 0
            and source_count >= self.batch_config.parallel_threshold
            and self.batch_config.max_workers > 1
        )

    def _discover_source_files(
        self,
        root: Path,
        *,
        count_excluded_files: bool = False,
    ) -> tuple[list[Path], int]:
        source_paths: list[Path] = []
        files_excluded = 0 if count_excluded_files else -1
        extensions = self.runtime.file_extensions

        for current_root, dir_names, file_names in os.walk(root):
            current_path = Path(current_root)
            excluded_dirs = [
                dir_name for dir_name in dir_names if self._is_excluded_dir(dir_name)
            ]
            if count_excluded_files:
                files_excluded += sum(
                    self._count_source_files(current_path / dir_name)
                    for dir_name in excluded_dirs
                )
            dir_names[:] = [
                dir_name for dir_name in dir_names if dir_name not in excluded_dirs
            ]

            for file_name in file_names:
                file_path = current_path / file_name
                if file_path.suffix.lower() in extensions:
                    source_paths.append(file_path.resolve())

        return sorted(source_paths), files_excluded

    def _count_source_files(self, root: Path) -> int:
        count = 0
        extensions = self.runtime.file_extensions
        for current_root, dir_names, file_names in os.walk(root):
            dir_names[:] = [
                dir_name
                for dir_name in dir_names
                if not self._is_excluded_dir(dir_name)
            ]
            count += sum(
                1
                for file_name in file_names
                if (Path(current_root) / file_name).suffix.lower() in extensions
            )
        return count

    def _is_excluded_dir(self, name: str) -> bool:
        return name in self.excluded_dir_names or name.startswith(".")

    def _extract_batch(
        self,
        root: Path,
        source_paths: list[Path],
    ) -> dict[FileId, SourceFile]:
        if not source_paths:
            return {}

        runtime = self.runtime
        if not runtime.script_path.is_file():
            raise RuntimeError(
                f"{runtime.language} extractor script is missing: {runtime.script_path}"
            )
        paths_by_source_id = {
            file_id_for_path(root, source_path): str(source_path)
            for source_path in source_paths
        }
        command = [
            *runtime.command,
            str(runtime.script_path),
            "--batch",
            str(root),
        ]
        if self.batch_config.output_format == "jsonl":
            command.append("--jsonl")

        completed = subprocess.run(
            command,
            input=json.dumps(paths_by_source_id),
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                f"{runtime.language} extractor failed with exit code "
                f"{completed.returncode}: {message}"
            )

        extracted = self._parse_batch_output(completed.stdout)
        source_paths_by_id = {
            file_id_for_path(root, source_path): source_path
            for source_path in source_paths
        }
        extracted_files: dict[FileId, SourceFile] = {}
        for raw_file_id, source_file in extracted.items():
            file_id = FileId(raw_file_id)
            source_path = source_paths_by_id[file_id]
            extracted_files[file_id] = SourceFile.model_validate(
                {
                    **source_file,
                    "file_id": file_id,
                    "module_id": module_id_for_path(root, source_path),
                }
            )
        return extracted_files

    @staticmethod
    def _merge_files(
        files: dict[FileId, SourceFile],
        extracted: dict[FileId, SourceFile],
    ) -> None:
        for file_id, source_file in extracted.items():
            if file_id in files:
                raise DuplicateIdentityError("file", file_id, file_id, file_id)
            files[file_id] = source_file

    def _parse_batch_output(self, output: str) -> dict[str, Any]:
        if self.batch_config.output_format == "jsonl":
            result: dict[str, Any] = {}
            for line in output.splitlines():
                if not line.strip():
                    continue
                item: Any = json.loads(line)
                if not isinstance(item, list):
                    raise RuntimeError("Extractor returned invalid JSONL batch output.")
                item_list = cast(list[Any], item)
                if len(item_list) != 2:
                    raise RuntimeError("Extractor returned invalid JSONL batch output.")
                source_id, source_file = item_list
                if not isinstance(source_id, str):
                    raise RuntimeError("Extractor returned a non-string source id.")
                result[source_id] = source_file
            return result

        parsed: Any = json.loads(output)
        if not isinstance(parsed, dict):
            raise RuntimeError("Extractor returned invalid JSON batch output.")
        return cast(dict[str, Any], parsed)

    def _source_id_for_path(self, root: Path, path: Path) -> FileId:
        return file_id_for_path(root, path)
