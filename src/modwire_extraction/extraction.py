from pathlib import Path

from .code import CodeMap, QueryableCodeMap
from .dependency import build_dependency_graph
from .extractors import languages


class ModwireExtraction:
    def __init__(self, root: Path):
        self._root = root.resolve()

    def discover(self) -> tuple[str, ...]:
        discovered: list[str] = []
        for language in languages.get_supported_languages():
            extractor = languages.load_extractor(language)
            if extractor.has_source_files(self._root):
                discovered.append(language)
        return tuple(discovered)

    def generate_map(
        self,
        language: str,
        *,
        count_excluded_files: bool = False,
    ) -> CodeMap:
        available = languages.get_supported_languages()
        if language not in available:
            raise ValueError(f"Language is not supported: {language}")

        extraction = languages.load_extractor(language).extract_source(
            self._root,
            count_excluded_files=count_excluded_files,
        )
        dependency_graph = build_dependency_graph(extraction.files)
        return CodeMap(
            language=language,
            extraction=extraction,
            dependency_graph=dependency_graph,
        )

    def generate_queryable_map(
        self,
        language: str,
        *,
        count_excluded_files: bool = False,
    ) -> QueryableCodeMap:
        code_map = self.generate_map(
            language,
            count_excluded_files=count_excluded_files,
        )
        return QueryableCodeMap(code_map=code_map)
