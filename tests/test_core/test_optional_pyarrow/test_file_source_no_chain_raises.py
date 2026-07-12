"""When ``ComputeFramework._materialize_descriptor`` finds no transformation chain, the
ValueError message is FULLY BACKEND-NEUTRAL.

mloda core has no required backend: pyarrow is an optional extra. So the "no transformer
chain" error never blames pyarrow. It simply states the missing transformer pair and the
generic remedy (register a transformer for that pair), identically whether or not pyarrow
happens to be installed.

The target framework here is a dummy type with no registered transformer, so a
``FileSource`` cannot be materialized into it via any chain and the "no chain" branch fires
regardless of whether pyarrow is installed. The pyarrow-availability seam is the module-
level ``pa`` symbol in ``mloda.core.abstract_plugins.compute_framework``; the tests toggle
it with monkeypatch instead of depending on the tox env, so they are portable across the
pyarrow and no-pyarrow envs and can pin the message under both states.

Contract:

  * The message never mentions pyarrow (neither ``pyarrow`` nor ``mloda[pyarrow]``), in
    either pyarrow state.
  * It names the source framework (``FileSource``) and carries the generic remedy
    (``Register`` a ``transformer`` for the pair).
  * The target is rendered by class name (``_to_fw.__name__`` when ``_to_fw`` is a class),
    not a raw ``<class '...'>`` repr, and a non-class target (``None``, the documented
    ``expected_data_framework`` default) still yields a ValueError, not an AttributeError.
"""

from __future__ import annotations

import pytest

import mloda.core.abstract_plugins.compute_framework as compute_framework
from mloda.core.abstract_plugins.components.input_data.file_source import FileSource
from mloda.provider import ComputeFramework
from mloda.user import ParallelizationMode


class _Unmaterializable:
    """A target type with no registered transformer: no FileSource chain can reach it."""


def _no_chain_framework() -> ComputeFramework:
    # Defined locally so the throwaway ComputeFramework subclass is not picked up by other
    # tests' framework enumeration.
    class _NoChainFramework(ComputeFramework):
        @classmethod
        def expected_data_framework(cls) -> type:
            return _Unmaterializable

    return _NoChainFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())


def test_no_chain_message_is_backend_neutral_without_pyarrow(monkeypatch: pytest.MonkeyPatch) -> None:
    """pyarrow absent: the message does NOT blame pyarrow; it states the missing pair and
    the generic remedy.
    """
    monkeypatch.setattr(compute_framework, "pa", None)

    fw = _no_chain_framework()
    source = FileSource(path="/nonexistent.csv", format="csv", columns=("A", "B"))

    with pytest.raises(ValueError) as excinfo:
        fw.transform(source, ["A", "B"])

    message = str(excinfo.value)
    assert "pyarrow" not in message.lower(), message
    assert "mloda[pyarrow]" not in message, message
    assert "FileSource" in message, message
    assert "Register" in message, message
    assert "transformer" in message, message


def test_no_chain_message_is_backend_neutral_with_pyarrow(monkeypatch: pytest.MonkeyPatch) -> None:
    """pyarrow present: the message is identical in spirit, still backend-neutral, naming
    the source framework and the generic remedy."""
    monkeypatch.setattr(compute_framework, "pa", object())

    fw = _no_chain_framework()
    source = FileSource(path="/nonexistent.csv", format="csv", columns=("A", "B"))

    with pytest.raises(ValueError) as excinfo:
        fw.transform(source, ["A", "B"])

    message = str(excinfo.value)
    assert "pyarrow" not in message.lower(), message
    assert "mloda[pyarrow]" not in message, message
    assert "FileSource" in message, message
    assert "Register" in message, message
    assert "transformer" in message, message
    assert _Unmaterializable.__name__ in message, message


def test_no_chain_renders_target_by_class_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """The target framework is rendered by ``__name__``, not a raw class repr."""
    monkeypatch.setattr(compute_framework, "pa", object())

    fw = _no_chain_framework()
    source = FileSource(path="/nonexistent.csv", format="csv", columns=("A", "B"))

    with pytest.raises(ValueError) as excinfo:
        fw.transform(source, ["A", "B"])

    message = str(excinfo.value)
    assert _Unmaterializable.__name__ in message, message
    assert "<class" not in message, message


def _none_target_framework() -> ComputeFramework:
    # A framework that leaves expected_data_framework() at its documented default (None).
    # Defined locally, distinct from _NoChainFramework, so it is not picked up by other
    # tests' framework enumeration.
    class _NoneTargetFramework(ComputeFramework):
        pass

    return _NoneTargetFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())


def test_none_target_framework_raises_value_error_not_attribute_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A framework whose expected_data_framework() returns a non-type value (None, the
    documented default) still raises ValueError, not AttributeError, and still names the
    source framework.
    """
    monkeypatch.setattr(compute_framework, "pa", object())

    fw = _none_target_framework()
    source = FileSource(path="/nonexistent.csv", format="csv", columns=("A", "B"))

    with pytest.raises(ValueError) as excinfo:
        fw.transform(source, ["A", "B"])

    assert "FileSource" in str(excinfo.value), str(excinfo.value)
