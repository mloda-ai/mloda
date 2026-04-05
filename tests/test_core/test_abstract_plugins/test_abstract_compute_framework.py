import pytest
from uuid import uuid4

from mloda.provider import ComputeFramework
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode


class BaseTestComputeFramework1(ComputeFramework):
    pass


class BaseTestComputeFramework2(ComputeFramework):
    pass


class BaseTestComputeFramework3(ComputeFramework):
    pass


class TestComputeFrameworkValidationErrors:
    def test_validate_expected_framework_includes_expected_type(self) -> None:
        """The error message should contain 'Expected:' and the expected data framework type name.

        Currently the message only says the data type is not supported, without telling the user
        what type was expected. The improved message should include 'Expected:' followed by the
        expected framework type.
        """

        class DictFramework(ComputeFramework):
            @classmethod
            def expected_data_framework(cls) -> type:
                return dict

        fw = DictFramework(
            mode=ParallelizationMode.SYNC,
            children_if_root=frozenset(),
            uuid=uuid4(),
        )
        fw.data = "this is a string, not a dict"

        with pytest.raises(ValueError, match=r"Expected type:"):
            fw.validate_expected_framework()
