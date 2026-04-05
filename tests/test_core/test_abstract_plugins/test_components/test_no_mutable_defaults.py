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
        d1.add_file("x.csv")
        assert "x.csv" not in d2.files

    def test_folders_not_shared(self) -> None:
        d1 = DataAccessCollection()
        d2 = DataAccessCollection()
        d1.add_folder("/data/folder")
        assert "/data/folder" not in d2.folders

    def test_initialized_connections_not_shared(self) -> None:
        d1 = DataAccessCollection()
        d2 = DataAccessCollection()
        d1.add_initialized_connection_object("conn1")
        assert "conn1" not in d2.initialized_connection_objects

    def test_uninitialized_connections_not_shared(self) -> None:
        d1 = DataAccessCollection()
        d2 = DataAccessCollection()
        d1.add_uninitialized_connection_object("conn1")
        assert "conn1" not in d2.uninitialized_connection_objects

    def test_credential_dicts_not_shared(self) -> None:
        d1 = DataAccessCollection()
        d2 = DataAccessCollection()
        d1.add_credential_dict({"user": "admin"})
        assert d2.credential_dicts != d1.credential_dicts

    def test_explicit_values_still_work(self) -> None:
        d = DataAccessCollection(
            files={"a.csv"},
            folders={"/data"},
            credential_dicts={"key": "val"},
            initialized_connection_objects={"conn"},
            uninitialized_connection_objects=["obj"],
        )
        assert "a.csv" in d.files
        assert "/data" in d.folders
        assert "conn" in d.initialized_connection_objects
        assert "obj" in d.uninitialized_connection_objects
