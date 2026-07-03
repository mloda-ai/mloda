"""Pins for hoisting init_reader into BaseInputData as the single concrete implementation.

Planned contract:

    BaseInputData.init_reader(self, options: Optional[Options]) -> tuple[BaseInputData, Any]
    becomes concrete, and ReadDB, ReadFile, and ReadDocument DELETE their overrides.

    Semantics of the hoisted implementation:
      - options is None: ValueError whose message starts
        "Options were not set for {self.__class__.__name__}" and contains "BaseInputData",
        "Options(context=", and an example line containing "ReaderClass" and "data_access".
      - options.get("BaseInputData") is None (key missing or None): ValueError containing
        "'BaseInputData' key is missing or None", the class name, and the same example hints.
      - happy path: options carrying {"BaseInputData": (SomeReaderClass, data_access)}
        returns (instance of SomeReaderClass, data_access).

Current-state notes (why the sharing pins fail today):
      - BaseInputData.init_reader is a bare "raise NotImplementedError".
      - ReadDB, ReadFile, and ReadDocument each define their own init_reader, so
        cls.init_reader is not BaseInputData.init_reader for any of them.
      - Options bracket access delegates to Options.get, so a missing key yields None
        (never KeyError); ReadDB's options["BaseInputData"] already funnels a missing
        key into its None branch today.

Test isolation note:
All BaseInputData subclasses used here are defined at MODULE scope, never inside test
methods, so they are picklable and stable in the global subclass registry. They leave
every matching-related classmethod (suffix, is_valid_credentials, load_data hooks, ...)
unimplemented, so supports_scoped_data_access() is False and plugin discovery in sibling
tests can never select them (mirrors test_read_document_load_data_seam.py).
"""

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda.core.abstract_plugins.components.options import Options
from mloda_plugins.feature_group.input_data.read_db import ReadDB
from mloda_plugins.feature_group.input_data.read_document import ReadDocument
from mloda_plugins.feature_group.input_data.read_file import ReadFile


class HoistBareInputData(BaseInputData):
    """Minimal BaseInputData subclass implementing nothing; drives the base init_reader directly."""


class HoistSentinelReader(BaseInputData):
    """Reader class carried inside the 'BaseInputData' options tuple for the happy-path pin."""


class HoistDocumentReader(ReadDocument):
    """ReadDocument family member; pins the message upgrade over the current terse wording."""


class TestInitReaderIsHoistedToBase:
    """After the hoist, the three reader families no longer override init_reader."""

    @pytest.mark.parametrize("family", [ReadDB, ReadFile, ReadDocument])
    def test_family_does_not_override_init_reader(self, family: type[BaseInputData]) -> None:
        # init_reader is a plain instance method; accessing it on the class object yields
        # the plain function, so identity comparison works without __func__ unwrapping.
        assert family.init_reader is BaseInputData.init_reader


class TestBaseInitReaderIsConcrete:
    """BaseInputData.init_reader becomes the single concrete implementation."""

    def test_options_none_raises_value_error_not_not_implemented(self) -> None:
        reader = HoistBareInputData()
        with pytest.raises(ValueError) as excinfo:
            reader.init_reader(None)
        message = str(excinfo.value)
        assert "HoistBareInputData" in message
        assert "BaseInputData" in message

    def test_options_none_message_contract(self) -> None:
        reader = HoistBareInputData()
        with pytest.raises(ValueError) as excinfo:
            reader.init_reader(None)
        message = str(excinfo.value)
        assert message.startswith("Options were not set for HoistBareInputData")
        assert "Options(context=" in message
        assert "ReaderClass" in message
        assert "data_access" in message

    def test_missing_key_message_contract(self) -> None:
        reader = HoistBareInputData()
        options = Options(context={"other_key": "value"})
        with pytest.raises(ValueError) as excinfo:
            reader.init_reader(options)
        message = str(excinfo.value)
        assert "'BaseInputData' key is missing or None" in message
        assert "HoistBareInputData" in message
        assert "ReaderClass" in message
        assert "data_access" in message

    def test_happy_path_returns_reader_instance_and_data_access(self) -> None:
        data_access: dict[str, Any] = {"dsn": "x"}
        options = Options(group={"BaseInputData": (HoistSentinelReader, data_access)})
        reader, returned_data_access = HoistBareInputData().init_reader(options)
        assert isinstance(reader, HoistSentinelReader)
        assert returned_data_access is data_access


class TestFamilyBehaviorAfterHoist:
    """The families keep (or gain) the parameterized error messages through the shared base."""

    @pytest.mark.parametrize(
        "family, expected_name",
        [
            (ReadDB, "ReadDB"),
            (ReadFile, "ReadFile"),
            (HoistDocumentReader, "HoistDocumentReader"),
        ],
    )
    def test_options_none_mentions_class_name(self, family: type[BaseInputData], expected_name: str) -> None:
        reader = family()
        with pytest.raises(ValueError, match="Options were not set") as excinfo:
            reader.init_reader(None)
        assert expected_name in str(excinfo.value)

    def test_read_db_missing_key_matches_base_contract(self) -> None:
        # Options bracket access returns None for a missing key, so even ReadDB's current
        # bracket-access implementation must funnel a keyless Options into the same
        # ValueError as the base (never KeyError or a None-unpack TypeError).
        options = Options(context={"other_key": "value"})
        with pytest.raises(ValueError, match="'BaseInputData' key is missing or None"):
            ReadDB().init_reader(options)
