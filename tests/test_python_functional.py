from pathlib import Path
from unittest.mock import patch

import pytest

from modwire_extraction import ModwireExtraction
from modwire_extraction.code import CodeMap
from modwire_extraction.extractors.languages import load_extractor
from modwire_extraction.extractors.languages.base import (
    BatchConfig,
    ExtractorRuntime,
    SourceExtractor,
)
from modwire_extraction.extractors.source import SourceFile
from modwire_extraction.identity import (
    DuplicateIdentityError,
    FileId,
    file_id_for_path,
    module_id_for_path,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures"

SOURCE_FILE_SHAPE = set(SourceFile.model_fields)


class RecordingExtractor(SourceExtractor):
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    @property
    def runtime(self) -> ExtractorRuntime:
        return ExtractorRuntime(
            language="recording",
            file_extensions=(".recording",),
            command=("unused",),
            script_path=Path(__file__),
        )

    @property
    def batch_config(self) -> BatchConfig:
        return BatchConfig(
            size=5,
            parallel_threshold=3,
            parallel_size=2,
            max_workers=2,
        )

    def _extract_batch(
        self,
        root: Path,
        source_paths: list[Path],
    ) -> dict[FileId, SourceFile]:
        self.batch_sizes.append(len(source_paths))
        return {
            self._source_id_for_path(root, source_path): SourceFile(
                file_id=file_id_for_path(root, source_path),
                module_id=module_id_for_path(root, source_path),
                imports=[],
                exports=[],
                classes=[],
                interfaces=[],
                types=[],
                abstract_classes=[],
                functions=[],
                values=[],
                callables=[],
                calls=[],
                line_count=1,
                code_line_count=1,
                public_symbol_count=0,
            )
            for source_path in source_paths
        }


def _language_roots() -> list[Path]:
    return sorted(path for path in FIXTURE_ROOT.iterdir() if path.is_dir())


def _skip_missing_runtime(language: str) -> None:
    try:
        load_extractor(language)
    except RuntimeError as error:
        if "extractor runtime is not available on PATH" in str(error):
            pytest.skip(str(error))
        raise


@pytest.mark.parametrize("root", _language_roots(), ids=lambda path: path.name)
def test_public_api_reads_each_language_project_with_same_shape(root: Path) -> None:
    language = root.name
    _skip_missing_runtime(language)

    extraction = ModwireExtraction(root)
    assert extraction.discover() == (language,)

    queryable_map = extraction.generate_queryable_map(language)
    code_map = queryable_map.code_map
    files_dict = code_map.extraction.files_dict()

    assert queryable_map.cm is code_map
    assert code_map.language == language
    assert set(files_dict) == set(code_map.extraction.files)
    assert code_map.extraction.files_found == len(code_map.extraction.files)
    assert code_map.extraction.files_excluded == -1
    assert all(
        set(type(source_file).model_fields) == SOURCE_FILE_SHAPE
        for source_file in files_dict.values()
    )
    assert all(source_file.file_id == file_id for file_id, source_file in files_dict.items())
    assert set(code_map.extraction.modules.values()) == set(files_dict)
    assert all(
        imported.resolution in {"resolved", "external"}
        for source_file in files_dict.values()
        for imported in source_file.imports
    )

    controller = (
        queryable_map.source_files()
        .where_contains(lambda result: result.source_id, "interfaces/http/controller")
        .first()
    )

    assert controller is not None
    assert controller.source_id in files_dict
    assert controller.file == files_dict[controller.source_id]

    source_files = (
        queryable_map.source_files()
        .where(lambda result: result.source_id.startswith("src/"))
        .all()
    )

    assert len(source_files) == code_map.extraction.files_found
    assert (
        queryable_map.query(files_dict.items()).where(_has_public_symbols).count()
        >= 1
    )


def test_queryable_code_map_exposes_report_query_surfaces() -> None:
    queryable_map = ModwireExtraction(FIXTURE_ROOT / "python").generate_queryable_map(
        "python"
    )

    assert queryable_map.source_ids() == (
        "src/application/use_cases/activate.py",
        "src/domain/model/user.py",
        "src/domain/services/policy.py",
        "src/interfaces/http/controller.py",
    )
    assert queryable_map.has_source_file("src/interfaces/http/controller.py")
    assert queryable_map.files().count() == queryable_map.source_files().count()

    controller = queryable_map.source_file("src/interfaces/http/controller.py")
    assert controller is not None
    assert controller.file.classes[0].name == "ActivationController"

    controller_class = (
        queryable_map.classes()
        .where_equal(lambda result: result.item.name, "ActivationController")
        .first()
    )
    assert controller_class is not None
    assert controller_class.source_id == "src/interfaces/http/controller.py"

    activation_label = (
        queryable_map.functions()
        .where_equal(lambda result: result.item.name, "activation_label")
        .first()
    )
    assert activation_label is not None
    assert activation_label.source_id == "src/application/use_cases/activate.py"

    domain_model_imports = queryable_map.imports().where_equal(
        lambda result: result.item.normalized_path,
        "domain/model/user",
    )
    assert domain_model_imports.count() == 3
    assert queryable_map.exports().count() == 8

    policy_method = (
        queryable_map.callables()
        .where_equal(lambda result: result.item.qualified_name, "ActivationPolicy.allows")
        .first()
    )
    assert policy_method is not None
    assert policy_method.source_id == "src/domain/services/policy.py"

    resolved_call = (
        queryable_map.calls()
        .where_equal(lambda result: result.item.resolution, "resolved")
        .first()
    )
    assert resolved_call is not None
    assert (
        resolved_call.item.target_callable_id
        == "src/domain/services/policy.py::can_activate"
    )

    controller_activation_edges = queryable_map.dependencies_between(
        "src/interfaces/http/controller.py",
        "src/application/use_cases/activate.py",
    )
    assert controller_activation_edges.count() == 2
    assert (
        queryable_map.outgoing_dependencies("src/interfaces/http/controller.py").count()
        == 4
    )
    assert queryable_map.incoming_dependencies("src/domain/model/user.py").count() == 3
    assert queryable_map.dependency_edges().count() == 10
    assert queryable_map.tracked_dependency_edges().count() == 7
    assert queryable_map.external_dependency_edges().count() == 3

    source_node = (
        queryable_map.dependency_nodes()
        .where_equal(lambda result: result.node_id, "src/interfaces/http/controller.py")
        .first()
    )
    assert source_node is not None
    assert source_node.file == controller.file

    external_edge = (
        queryable_map.external_dependency_edges()
        .where_equal(lambda result: result.edge.specifier, "json")
        .first()
    )
    assert external_edge is not None
    assert external_edge.edge.resolution == "external"
    assert external_edge.edge.to_id is None


def test_discover_ignores_excluded_source_directories(tmp_path: Path) -> None:
    ignored_root = tmp_path / "ignored"
    ignored_root.mkdir()
    (ignored_root / "generated.py").write_text("def generated():\n    return None\n")

    assert ModwireExtraction(tmp_path).discover() == ()


def test_excluded_source_directories_are_not_traversed_by_default(
    tmp_path: Path,
) -> None:
    excluded_root = tmp_path / "node_modules"
    excluded_root.mkdir()
    (excluded_root / "ignored.py").write_text("value = 1\n")
    (tmp_path / "included.py").write_text("value = 1\n")

    extraction = ModwireExtraction(tmp_path)
    with patch.object(
        SourceExtractor,
        "_count_source_files",
        side_effect=AssertionError("excluded tree was traversed"),
    ):
        assert extraction.discover() == ("python",)
        code_map = extraction.generate_map("python")

    assert set(code_map.extraction.files) == {"included.py"}
    assert code_map.extraction.files_excluded == -1


def test_excluded_source_file_count_can_be_requested_explicitly(
    tmp_path: Path,
) -> None:
    excluded_root = tmp_path / "node_modules"
    nested_root = excluded_root / "dependency"
    nested_root.mkdir(parents=True)
    (nested_root / "ignored.py").write_text("value = 1\n")
    (tmp_path / "included.py").write_text("value = 1\n")

    code_map = ModwireExtraction(tmp_path).generate_map(
        "python",
        count_excluded_files=True,
    )

    assert set(code_map.extraction.files) == {"included.py"}
    assert code_map.extraction.files_excluded == 1


def test_source_extractor_uses_parallel_batch_config(tmp_path: Path) -> None:
    for index in range(5):
        (tmp_path / f"{index}.recording").write_text("source\n")

    extractor = RecordingExtractor()
    extraction = extractor.extract_source(tmp_path)

    assert extraction.files_found == 5
    assert set(extraction.files) == {
        "0.recording",
        "1.recording",
        "2.recording",
        "3.recording",
        "4.recording",
    }
    assert sorted(extractor.batch_sizes) == [1, 2, 2]


def test_python_extraction_raises_on_syntax_error_files(tmp_path: Path) -> None:
    source_path = tmp_path / "broken.py"
    source_path.write_text("1syntax_error\n")

    with pytest.raises(RuntimeError, match="python extractor failed with exit code"):
        ModwireExtraction(tmp_path).generate_map("python")


def test_typescript_extraction_raises_on_syntax_error_files(tmp_path: Path) -> None:
    _skip_missing_runtime("typescript")
    source_path = tmp_path / "broken.ts"
    source_path.write_text("export const = ;\n")

    with pytest.raises(RuntimeError, match="typescript extractor failed with exit code"):
        ModwireExtraction(tmp_path).generate_map("typescript")


def test_typescript_extraction_tolerates_non_literal_import_specifiers(
    tmp_path: Path,
) -> None:
    _skip_missing_runtime("typescript")
    source_path = tmp_path / "broken.ts"
    source_path.write_text(
        "import { value } from moduleName;\n"
        "for (const fn = () => value; false;) {}\n"
        "export const ok = 1;\n"
    )

    code_map = ModwireExtraction(tmp_path).generate_map("typescript")

    assert code_map.extraction.files_found == 1
    assert code_map.extraction.files["broken.ts"].imports == []
    assert code_map.extraction.files["broken.ts"].public_symbol_count == 1


def test_php_extraction_tolerates_anonymous_class_construction(
    tmp_path: Path,
) -> None:
    _skip_missing_runtime("php")
    source_path = tmp_path / "anonymous.php"
    source_path.write_text(
        "<?php\n"
        "class Factory {\n"
        "    public function build(): object {\n"
        "        return new class {};\n"
        "    }\n"
        "}\n"
    )

    code_map = ModwireExtraction(tmp_path).generate_map("php")

    assert code_map.extraction.files_found == 1
    assert code_map.extraction.files["anonymous.php"].classes[0].name == "Factory"
    assert code_map.extraction.files["anonymous.php"].calls == []


def test_php_extraction_raises_on_syntax_error_files(tmp_path: Path) -> None:
    _skip_missing_runtime("php")
    source_path = tmp_path / "broken.php"
    source_path.write_text("<?php\nfunction broken(\n")

    with pytest.raises(RuntimeError, match="php extractor failed with exit code"):
        ModwireExtraction(tmp_path).generate_map("php")


def test_missing_external_runtime_raises_stable_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "src"
    source_root.mkdir()
    (source_root / "controller.ts").write_text("export const value = 1;\n")
    monkeypatch.setattr(
        "modwire_extraction.extractors.languages.loader.shutil.which",
        lambda executable: None,
    )

    with pytest.raises(
        RuntimeError,
        match="typescript extractor runtime is not available on PATH: node",
    ):
        ModwireExtraction(tmp_path).generate_map("typescript")


@pytest.mark.parametrize("root", _language_roots(), ids=lambda path: path.name)
def test_code_map_serialization_round_trips_through_pydantic(root: Path) -> None:
    language = root.name
    _skip_missing_runtime(language)

    original = ModwireExtraction(root).generate_map(language)
    payload = original.model_dump(mode="python")

    assert set(payload) == {"language", "extraction", "dependency_graph"}
    assert set(payload["extraction"]) == {
        "files",
        "modules",
        "files_found",
        "files_excluded",
    }
    assert set(payload["dependency_graph"]) == {"nodes", "edges"}

    restored = CodeMap.model_validate(payload)
    json_restored = CodeMap.model_validate_json(original.model_dump_json())

    assert restored.model_dump(mode="python") == payload
    assert json_restored.model_dump(mode="python") == payload
    assert restored.dependency_graph.node_ids() == original.dependency_graph.node_ids()


def test_same_stem_typescript_files_report_module_identity_collision(
    tmp_path: Path,
) -> None:
    _skip_missing_runtime("typescript")
    (tmp_path / "component.ts").write_text("export const component = 1;\n")
    (tmp_path / "component.tsx").write_text("export const view = <div />;\n")

    with pytest.raises(DuplicateIdentityError) as raised:
        ModwireExtraction(tmp_path).generate_map("typescript")

    assert raised.value.as_dict() == {
        "code": "duplicate_identity",
        "identity_kind": "module",
        "identity": "component",
        "existing_file_id": "component.ts",
        "duplicate_file_id": "component.tsx",
    }


def test_unresolved_relative_import_remains_queryable(tmp_path: Path) -> None:
    _skip_missing_runtime("typescript")
    (tmp_path / "entry.ts").write_text(
        'import { missing } from "./missing";\nexport { missing };\n'
    )

    code_map = ModwireExtraction(tmp_path).generate_map("typescript")
    imported = code_map.extraction.files["entry.ts"].imports[0]
    edge = code_map.dependency_graph.edges[0]

    assert imported.path == "./missing"
    assert imported.normalized_path == "missing"
    assert imported.resolution == "unresolved"
    assert imported.target_file_id is None
    assert edge.resolution == "unresolved"
    assert edge.specifier == "missing"
    assert edge.to_id is None


@pytest.mark.parametrize(
    ("language", "resolved", "external"),
    (
        ("python", 7, 3),
        ("typescript", 5, 2),
        ("php", 6, 2),
    ),
)
def test_cross_language_dependency_resolution(
    language: str,
    resolved: int,
    external: int,
) -> None:
    _skip_missing_runtime(language)
    graph = ModwireExtraction(FIXTURE_ROOT / language).generate_map(
        language
    ).dependency_graph

    assert sum(edge.resolution == "resolved" for edge in graph.edges) == resolved
    assert sum(edge.resolution == "external" for edge in graph.edges) == external
    assert all(
        (edge.to_id is not None) == (edge.resolution == "resolved")
        for edge in graph.edges
    )


def _has_public_symbols(item: tuple[str, SourceFile]) -> bool:
    return item[1].public_symbol_count > 0
