"""
Regression guard for issue #828.

Doc-example snippets are executed in-process, so feature resolution sees every
FeatureGroup subclass alive in the pytest worker. A test-only data creator that
exposes "customer_id" then collides with ApiInputDataFeature when a doc snippet
requests the bare root feature "customer_id" via api_data. This test simulates
that leak and requires the doc snippet check to be unaffected by it.
"""

import gc
import textwrap
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from mloda.provider import BaseInputData, DataCreator, FeatureGroup, FeatureSet
from tests.test_documentation.test_documentation import run_md_file_isolated

DOC_SNIPPET_MD = textwrap.dedent(
    """\
    # Doc example

    ```python
    from mloda.user import PluginLoader, mloda

    PluginLoader.all()

    result = mloda.run_all(
        features=["customer_id"],
        compute_frameworks=["PandasDataFrame"],
        api_data={
            "SampleData": {
                "customer_id": ["C001", "C002"],
            }
        },
    )
    ```
    """
)


@pytest.fixture
def leaked_class_registry() -> Generator[list[type[FeatureGroup]], None, None]:
    registry: list[type[FeatureGroup]] = []
    yield registry
    # FeatureGroup.__subclasses__ holds weak references; dropping the last strong
    # reference and collecting removes the polluting class from the resolution universe.
    registry.clear()
    gc.collect()


@pytest.mark.timeout(60)
def test_doc_snippets_unaffected_by_leaked_test_feature_groups(
    tmp_path: Path, leaked_class_registry: list[type[FeatureGroup]]
) -> None:
    class LeakedCustomerIdDataCreator(FeatureGroup):
        @classmethod
        def input_data(cls) -> BaseInputData | None:
            return DataCreator({"customer_id"})

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return {"customer_id": ["C001", "C002"]}

    leaked_class_registry.append(LeakedCustomerIdDataCreator)
    # Drop the local name so a failure traceback frame cannot keep the class
    # alive past the fixture cleanup; the registry holds the only strong reference.
    del LeakedCustomerIdDataCreator

    md_file = tmp_path / "leaked_customer_id_doc.md"
    md_file.write_text(DOC_SNIPPET_MD, encoding="utf-8")

    run_md_file_isolated(md_file)
