"""Tests that enforce consistent usage of CHAIN_SEPARATOR across experimental plugins.

These tests detect two categories of violations:
1. Dead-code PATTERN = "__" class attributes that are never read by framework code.
2. Hardcoded "__" string literals in method bodies that should use CHAIN_SEPARATOR.
"""

import ast
import pathlib
import re
from typing import List


PLUGIN_DIR = pathlib.Path("mloda_plugins/feature_group/experimental")


def _collect_python_files() -> List[pathlib.Path]:
    """Collect all Python files under the experimental plugins directory."""
    return sorted(PLUGIN_DIR.rglob("*.py"))


class TestNoPatternAttribute:
    """No plugin should define a class-level PATTERN = "__" attribute.

    The PATTERN attribute is dead code: it is never read by any framework
    code. Plugins that need the chain separator should import and use
    CHAIN_SEPARATOR from mloda.provider instead.
    """

    def test_no_class_level_pattern_attribute(self) -> None:
        """Scan all experimental plugin files for class-level PATTERN = '__' assignments."""
        violations = []

        for py_file in _collect_python_files():
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))

            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue

                for child in node.body:
                    if not isinstance(child, ast.Assign):
                        continue

                    for target in child.targets:
                        if not isinstance(target, ast.Name):
                            continue
                        if target.id != "PATTERN":
                            continue

                        # Check if the value is the string "__"
                        value = child.value
                        if isinstance(value, ast.Constant) and value.value == "__":
                            violations.append(
                                f"{py_file}:{child.lineno} - "
                                f"class {node.name} defines dead-code "
                                f'PATTERN = "__" attribute'
                            )

        assert violations == [], (
            "Found class-level PATTERN = '__' attributes that are dead code. "
            "Remove them; plugins should use CHAIN_SEPARATOR from "
            "mloda.provider instead.\n\nViolations:\n" + "\n".join(violations)
        )


class TestNoHardcodedChainSeparator:
    """No plugin should hardcode "__" string literals in method bodies.

    All uses of the chain separator should go through the CHAIN_SEPARATOR
    constant imported from mloda.provider.
    """

    # Lines matching any of these patterns are allowed to contain "__".
    _ALLOWED_LINE_PATTERNS = [
        re.compile(r"^\s*(PREFIX_PATTERN|SUFFIX_PATTERN)\s*="),  # regex pattern assignments
        re.compile(r"^\s*CHAIN_SEPARATOR\s*="),  # the constant definition itself
        re.compile(r"^\s*#"),  # comment lines
        re.compile(r"^\s*(from|import)\s"),  # import lines
        re.compile(r"^\s*PATTERN\s*="),  # PATTERN assignments (caught by the other test)
    ]

    def test_no_hardcoded_double_underscore_literals(self) -> None:
        """Scan experimental plugin source lines for hardcoded '__' string literals."""
        violations = []

        for py_file in _collect_python_files():
            source = py_file.read_text(encoding="utf-8")
            lines = source.splitlines()

            in_docstring = False
            for lineno_0, line in enumerate(lines, start=1):
                stripped = line.strip()

                # Track docstring regions (triple-quoted strings).
                # Count triple-quote occurrences to toggle docstring state.
                triple_double = stripped.count('"""')
                triple_single = stripped.count("'''")
                triple_count = triple_double + triple_single

                if triple_count == 1:
                    # Opening or closing a docstring
                    in_docstring = not in_docstring
                    continue
                elif triple_count >= 2:
                    # Docstring opens and closes on the same line
                    continue

                if in_docstring:
                    continue

                # Skip lines matching allowed patterns
                if any(pat.search(line) for pat in self._ALLOWED_LINE_PATTERNS):
                    continue

                # Check for standalone "__" string literal (quoted)
                # Match both single and double quoted variants
                if '"__"' in line or "'__'" in line:
                    violations.append(f"{py_file}:{lineno_0} - hardcoded '__' literal: {stripped}")

        assert violations == [], (
            "Found hardcoded '__' string literals in plugin code. "
            "Use CHAIN_SEPARATOR from mloda.provider instead.\n\n"
            "Violations:\n" + "\n".join(violations)
        )
