"""Unit tests for canonical_dump in mloda.core.abstract_plugins.components.code_closure.

Pins the AST canonicalization rules: docstrings/comments/formatting are
ignored, code changes are not, strings are hex-encoded (never repr'd), and
large/pathological inputs do not raise.
"""

import ast

import pytest

from mloda.core.abstract_plugins.components.code_closure import canonical_dump


def _dump(src: str) -> str:
    return canonical_dump(ast.parse(src))


class TestDocstringsAndCommentsIgnored:
    def test_module_docstring_ignored(self) -> None:
        assert _dump('"""module doc"""\nx = 1\n') == _dump("x = 1\n")

    def test_class_docstring_ignored(self) -> None:
        src_a = "class C:\n    '''doc'''\n    x = 1\n"
        src_b = "class C:\n    x = 1\n"
        assert _dump(src_a) == _dump(src_b)

    def test_function_docstring_ignored(self) -> None:
        src_a = "def f():\n    '''doc'''\n    return 1\n"
        src_b = "def f():\n    return 1\n"
        assert _dump(src_a) == _dump(src_b)

    def test_comment_ignored(self) -> None:
        src_a = "x = 1  # a comment\n"
        src_b = "x = 1\n"
        assert _dump(src_a) == _dump(src_b)

    @pytest.mark.parametrize(
        ("src_a", "src_b"),
        [
            ("x = 1\n", "x = 2\n"),
            ("x = 1 + 1\n", "x = 1 - 1\n"),
            ("x = 1\n", "y = 1\n"),
        ],
    )
    def test_code_change_changes_dump(self, src_a: str, src_b: str) -> None:
        assert _dump(src_a) != _dump(src_b)


class TestBareStringStatementsDropped:
    def test_every_bare_string_statement_dropped_not_only_first(self) -> None:
        src_a = "def f():\n    'first bare string'\n    'second bare string'\n    return 1\n"
        src_b = "def f():\n    return 1\n"
        assert _dump(src_a) == _dump(src_b)


class TestKwDefaultsPlaceholder:
    def test_none_default_then_default_differs_from_default_then_none(self) -> None:
        dump_a = _dump("def f(*, a, b=1): pass\n")
        dump_b = _dump("def f(*, a=1, b): pass\n")
        assert dump_a != dump_b


class TestFormattingIgnored:
    @pytest.mark.parametrize(
        ("src_a", "src_b"),
        [
            ("x = 1\n\n\ny = 2\n", "x = 1\ny = 2\n"),
            ("x=1\ny  =  2\n", "x = 1\ny = 2\n"),
            ("x = (\n    1\n    + 2\n)\n", "x = 1 + 2\n"),
        ],
    )
    def test_formatting_variants_dump_equal(self, src_a: str, src_b: str) -> None:
        assert _dump(src_a) == _dump(src_b)

    def test_no_position_attributes_in_output(self) -> None:
        dump = _dump("x = 1\n")
        for token in ("lineno", "col_offset", "end_lineno", "end_col_offset"):
            assert token not in dump


class TestNonAsciiAndSurrogates:
    def test_lone_surrogate_constant_does_not_raise(self) -> None:
        node = ast.Constant(value="\ud800")
        ast.fix_missing_locations(node)
        canonical_dump(node)

    def test_string_dump_not_repr_and_matches_hex_rule(self) -> None:
        value = "꟎"
        expected = "s:" + value.encode("utf-8", "surrogatepass").hex()
        node = ast.Assign(
            targets=[ast.Name(id="x", ctx=ast.Store())],
            value=ast.Constant(value=value),
        )
        module = ast.Module(body=[node], type_ignores=[])
        ast.fix_missing_locations(module)
        dump = canonical_dump(module)
        assert expected in dump
        assert "\\u" not in dump
        assert "\\x" not in dump


class TestLargeInputsDoNotRaise:
    def test_int_literal_over_4300_digits_does_not_raise(self) -> None:
        node = ast.Constant(value=10**5000 - 1)
        ast.fix_missing_locations(node)
        canonical_dump(node)

    def test_flat_expression_2000_terms_does_not_raise_recursion_error(self) -> None:
        # 2000 terms parses on Python 3.10-3.14 and still exceeds the default
        # recursion limit, so a recursive dump would overflow too.
        src = "a" + " + a" * 1999
        tree = ast.parse(src, mode="eval")
        canonical_dump(tree)


class TestIdentifiersHexEncoded:
    def test_identifier_not_present_in_plain_text(self) -> None:
        dump = _dump("some_unusual_identifier_name = 1\n")
        assert "some_unusual_identifier_name" not in dump
