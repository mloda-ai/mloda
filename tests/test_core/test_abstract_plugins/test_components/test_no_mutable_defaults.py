"""Tests that mutable default arguments are not shared across instances."""

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature


class TestFeatureNoSharedMutableDefaults:
    def test_init_options_not_shared(self) -> None:
        f1 = Feature(name="a")
        f2 = Feature(name="b")
        assert f1.options is not f2.options

    def test_classmethod_not_typed_options_not_shared(self) -> None:
        f1 = Feature.not_typed("a")
        f2 = Feature.not_typed("b")
        assert f1.options is not f2.options

    def test_classmethod_str_of_options_not_shared(self) -> None:
        f1 = Feature.str_of("a")
        f2 = Feature.str_of("b")
        assert f1.options is not f2.options

    def test_classmethod_int32_of_options_not_shared(self) -> None:
        f1 = Feature.int32_of("a")
        f2 = Feature.int32_of("b")
        assert f1.options is not f2.options

    def test_classmethod_int64_of_options_not_shared(self) -> None:
        f1 = Feature.int64_of("a")
        f2 = Feature.int64_of("b")
        assert f1.options is not f2.options

    def test_classmethod_float_of_options_not_shared(self) -> None:
        f1 = Feature.float_of("a")
        f2 = Feature.float_of("b")
        assert f1.options is not f2.options

    def test_classmethod_double_of_options_not_shared(self) -> None:
        f1 = Feature.double_of("a")
        f2 = Feature.double_of("b")
        assert f1.options is not f2.options

    def test_classmethod_boolean_of_options_not_shared(self) -> None:
        f1 = Feature.boolean_of("a")
        f2 = Feature.boolean_of("b")
        assert f1.options is not f2.options

    def test_classmethod_binary_of_options_not_shared(self) -> None:
        f1 = Feature.binary_of("a")
        f2 = Feature.binary_of("b")
        assert f1.options is not f2.options

    def test_classmethod_date_of_options_not_shared(self) -> None:
        f1 = Feature.date_of("a")
        f2 = Feature.date_of("b")
        assert f1.options is not f2.options

    def test_classmethod_timestamp_millis_of_options_not_shared(self) -> None:
        f1 = Feature.timestamp_millis_of("a")
        f2 = Feature.timestamp_millis_of("b")
        assert f1.options is not f2.options

    def test_classmethod_timestamp_micros_of_options_not_shared(self) -> None:
        f1 = Feature.timestamp_micros_of("a")
        f2 = Feature.timestamp_micros_of("b")
        assert f1.options is not f2.options

    def test_classmethod_decimal_of_options_not_shared(self) -> None:
        f1 = Feature.decimal_of("a")
        f2 = Feature.decimal_of("b")
        assert f1.options is not f2.options

    def test_explicit_options_still_work(self) -> None:
        opts = {"key": "value"}
        f = Feature(name="a", options=opts)
        assert f.options.get("key") == "value"

    def test_explicit_none_options_creates_empty(self) -> None:
        f = Feature(name="a", options=None)
        assert f.options is not None


class TestDataAccessCollectionNoSharedMutableDefaults:
    def test_files_not_shared(self) -> None:
        d1 = DataAccessCollection()
        d2 = DataAccessCollection()
        d1.add_file("f1", "x.csv")
        assert "f1" not in d2.files

    def test_folders_not_shared(self) -> None:
        d1 = DataAccessCollection()
        d2 = DataAccessCollection()
        d1.add_folder("d1", "/data/folder")
        assert "d1" not in d2.folders

    def test_connections_not_shared(self) -> None:
        d1 = DataAccessCollection()
        d2 = DataAccessCollection()
        d1.add_connection("conn1", "conn1")
        assert "conn1" not in d2.connections

    def test_credentials_not_shared(self) -> None:
        d1 = DataAccessCollection()
        d2 = DataAccessCollection()
        d1.add_credentials("c1", {"user": "admin"})
        assert d2.credentials != d1.credentials

    def test_explicit_values_still_work(self) -> None:
        d = DataAccessCollection(
            files={"f1": "a.csv"},
            folders={"d1": "/data"},
            credentials={"c1": {"key": "val"}},
            connections={"conn": "conn"},
        )
        assert "f1" in d.files
        assert "d1" in d.folders
        assert "conn" in d.connections
        assert "c1" in d.credentials
