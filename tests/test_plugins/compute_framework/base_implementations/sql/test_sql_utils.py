import pytest

from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import (
    inline_params,
    quote_ident,
    quote_value,
)


class TestQuoteIdent:
    def test_simple_name(self) -> None:
        assert quote_ident("col") == '"col"'

    def test_embedded_double_quote(self) -> None:
        assert quote_ident('col"name') == '"col""name"'

    def test_name_with_spaces(self) -> None:
        assert quote_ident("my col") == '"my col"'

    def test_name_with_special_chars(self) -> None:
        assert quote_ident("a.b") == '"a.b"'


class TestQuoteValue:
    def test_none(self) -> None:
        assert quote_value(None) == "NULL"

    def test_true(self) -> None:
        assert quote_value(True) == "1"

    def test_false(self) -> None:
        assert quote_value(False) == "0"

    def test_integer(self) -> None:
        assert quote_value(42) == "42"

    def test_negative_integer(self) -> None:
        assert quote_value(-5) == "-5"

    def test_float(self) -> None:
        assert quote_value(3.14) == "3.14"

    def test_string(self) -> None:
        assert quote_value("hello") == "'hello'"

    def test_string_with_single_quote(self) -> None:
        assert quote_value("it's") == "'it''s'"

    def test_string_with_sql_injection(self) -> None:
        assert quote_value("'; DROP TABLE users; --") == "'''; DROP TABLE users; --'"

    def test_inf_raises(self) -> None:
        with pytest.raises(ValueError):
            quote_value(float("inf"))

    def test_nan_raises(self) -> None:
        with pytest.raises(ValueError):
            quote_value(float("nan"))

    def test_neg_inf_raises(self) -> None:
        with pytest.raises(ValueError):
            quote_value(float("-inf"))

    def test_unknown_type_raises_type_error(self) -> None:
        """Unknown types must be rejected, not silently converted via str()."""

        class CustomObj:
            def __str__(self) -> str:
                return "injected"

        with pytest.raises(TypeError):
            quote_value(CustomObj())

    def test_bytes_raises_type_error(self) -> None:
        """bytes is not a supported SQL literal type."""
        with pytest.raises(TypeError):
            quote_value(b"hello")


class TestInlineParams:
    def test_single_param(self) -> None:
        result = inline_params("col = ?", ("hello",))
        assert result == "col = 'hello'"

    def test_multiple_params(self) -> None:
        result = inline_params("a = ? AND b = ?", (1, "x"))
        assert result == "a = 1 AND b = 'x'"

    def test_none_param(self) -> None:
        result = inline_params("col IS ?", (None,))
        assert result == "col IS NULL"

    def test_question_mark_inside_value_not_consumed(self) -> None:
        """A ? inside a substituted value must not be treated as the next placeholder."""
        result = inline_params("a = ? AND b = ?", ("a?b", "c"))
        assert result == "a = 'a?b' AND b = 'c'"

    def test_sql_injection_in_value(self) -> None:
        result = inline_params("col = ?", ("'; DROP TABLE t; --",))
        assert result == "col = '''; DROP TABLE t; --'"

    def test_no_params(self) -> None:
        result = inline_params("col = 1", ())
        assert result == "col = 1"

    def test_placeholder_count_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Placeholder count"):
            inline_params("a = ? AND b = ?", (1,))
