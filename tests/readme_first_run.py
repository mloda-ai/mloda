"""Run every README python block on a wheel install, as a new user would.

A block runs bare (offline wheel, no extras) unless the code block right before it, prose aside, is a
```bash block with ``pip install "mloda[<extras>]"``; then it runs in a venv holding the wheel plus those extras.
Usage: python -m tests.readme_first_run <README.md> <dist-dir>
"""

import os
import re
import subprocess  # nosec B404
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from tests.docs_corpus import OUTPUT_TAG, RUNNABLE_TAG

_FENCE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
_EXTRAS_INSTALL = re.compile(r"""pip install ["']?mloda\[([^\]]+)\]""")
_ABSENT_MODULES = ("pandas", "pyarrow", "numpy")
_PANDAS_HINT = 'pip install "mloda[pandas]"'
_BIN = "Scripts" if os.name == "nt" else "bin"
_timings: list[tuple[str, float]] = []


@dataclass(frozen=True)
class ReadmeBlock:
    """A README python block, the text block directly after it (None if absent) and the extras it needs."""

    code: str
    expected: str | None
    extras: tuple[str, ...]


def _extras(previous: "re.Match[str] | None") -> tuple[str, ...]:
    """Extras named by the code block before a python block: a bash ``pip install "mloda[...]"``, else none."""
    if previous is None or previous.group(1).strip() != "bash":
        return ()
    install = _EXTRAS_INSTALL.search(previous.group(2))
    if install is None:
        return ()
    return tuple(extra.strip() for extra in install.group(1).split(","))


def python_blocks(readme_text: str) -> list[ReadmeBlock]:
    """Every python block with the text block directly after it and its extras, in document order."""
    fences = list(_FENCE.finditer(readme_text))
    blocks: list[ReadmeBlock] = []
    for position, fence in enumerate(fences):
        if fence.group(1).strip() != RUNNABLE_TAG:
            continue
        expected = None
        following = fences[position + 1] if position + 1 < len(fences) else None
        if (
            following is not None
            and following.group(1).strip() == OUTPUT_TAG
            and not readme_text[fence.end() : following.start()].strip()
        ):
            expected = following.group(2)
        previous = fences[position - 1] if position else None
        blocks.append(ReadmeBlock(fence.group(2), expected, _extras(previous)))
    return blocks


def first_example(readme_text: str) -> tuple[str, str | None]:
    """The first python block and the text block directly after it (None if absent)."""
    blocks = python_blocks(readme_text)
    assert blocks, f"README has no ```{RUNNABLE_TAG} block"
    return blocks[0].code, blocks[0].expected


def _run(label: str | None, argv: list[str], env: dict[str, str] | None = None) -> "subprocess.CompletedProcess[str]":
    """Run argv; a label records the run in the time-to-first-success timings, None leaves it out."""
    start = time.monotonic()
    # Safe: argv is built from sys.executable, a temp venv and caller-given paths, no shell.
    result = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=300)  # nosec B603
    if label is not None:
        _timings.append((label, time.monotonic() - start))
    return result


def _clean_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if not k.startswith("MLODA_")}


def _summary() -> str:
    lines = [f"{seconds:7.2f}s  {label}" for label, seconds in _timings]
    lines.append(f"{sum(s for _, s in _timings):7.2f}s  total time to first success")
    return "\n".join(lines)


def _block_failure(index: int, code: str, expected: str | None, run: "subprocess.CompletedProcess[str]") -> str | None:
    """Why the run of README python block ``index`` is not acceptable, or None when it is."""
    first_line = next((line for line in code.splitlines() if line.strip()), "")
    where = f"README python block {index} ({first_line!r})"
    if run.returncode != 0:
        return f"{where} failed, exit code {run.returncode}\nstdout:\n{run.stdout}\nstderr:\n{run.stderr}"
    if run.stderr.strip():
        return f"{where} must run quietly, stderr was:\n{run.stderr}"
    if expected is not None and run.stdout.strip() != expected.strip():
        return f"{where} stdout differs from README\nexpected:\n{expected.strip()}\nactual:\n{run.stdout.strip()}"
    return None


def _venv(root: Path, wheel: Path, extras: tuple[str, ...]) -> str:
    """Create a venv under root with the wheel installed and return its python.

    The bare venv installs the wheel offline and is timed as time to first success; a venv with extras
    installs the same wheel plus the extras from the index and is not timed.
    """
    venv = root / ("venv-" + "-".join(extras) if extras else "venv")
    python = str(venv / _BIN / "python")
    start = time.monotonic()
    # uv, not `python -m venv`: distro Pythons often lack ensurepip.
    created = _run("uv venv" if not extras else None, ["uv", "venv", "--python", sys.executable, str(venv)])
    assert created.returncode == 0, f"{created.stdout}\n{created.stderr}"
    if extras:
        # PEP 508 direct reference: mloda from the local wheel, the extras from the index.
        requirement = f"mloda[{','.join(extras)}] @ {wheel.resolve().as_uri()}"
        install = _run(None, ["uv", "pip", "install", "--python", python, "--no-config", requirement])
        assert install.returncode == 0, f"{install.stdout}\n{install.stderr}"
        print(f"{time.monotonic() - start:7.2f}s  venv with {requirement}")
        return python
    install = _run(
        "uv pip install mloda (offline wheel, no extras)",
        ["uv", "pip", "install", "--python", python, "--no-index", "--no-config", str(wheel)],
    )
    assert install.returncode == 0, f"{install.stdout}\n{install.stderr}"
    probe = "import importlib.util as u; print([m for m in %r if u.find_spec(m)])" % (_ABSENT_MODULES,)
    # Safe: argv is the temp venv python and a constant script, no shell.
    probed = subprocess.run(  # nosec B603
        [python, "-I", "-c", probe], capture_output=True, text=True, env=_clean_env(), timeout=60
    )
    present = probed.stdout.strip()
    assert present == "[]", f"bare install must not contain optional backends, found {present}"
    return python


def main(readme: Path, dist: Path) -> int:
    blocks = python_blocks(readme.read_text(encoding="utf-8"))
    assert blocks, f"README has no ```{RUNNABLE_TAG} block"
    wheels = sorted(dist.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel in {dist}, found {wheels}"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # One venv per distinct extras tuple; the bare one first, the others when a block first needs them.
        pythons: dict[tuple[str, ...], str] = {(): _venv(root, wheels[0], ())}
        python = pythons[()]

        for index, block in enumerate(blocks, 1):
            if block.extras not in pythons:
                pythons[block.extras] = _venv(root, wheels[0], block.extras)
            script = root / f"block_{index}.py"
            script.write_text(block.code, encoding="utf-8")
            # Only the first block counts towards time to first success.
            label = "run README first example" if index == 1 else None
            run = _run(label, [pythons[block.extras], "-I", str(script)], env=_clean_env())
            if index == 1:
                print(_summary())
            failure = _block_failure(index, block.code, block.expected, run)
            if failure:
                print(failure)
                return 1
            if index == 1 and block.expected is None:
                print("README has no ```text block after the first example")
                return 1

        mloda_script = str(Path(python).with_name("mloda.exe" if os.name == "nt" else "mloda"))
        # Not recorded in the time-to-first-success timings.
        # Safe: argv is the temp venv's mloda console script and a fixed subcommand, no shell.
        demo = subprocess.run(  # nosec B603
            [mloda_script, "demo"], capture_output=True, text=True, env=_clean_env(), timeout=300
        )
        if demo.returncode != 0 or demo.stderr.strip():
            print(f"mloda demo failed, exit code {demo.returncode}\nstdout:\n{demo.stdout}\nstderr:\n{demo.stderr}")
            return 1
        missing = [s for s in ("ApiDataFG", "AggregatedFeatureGroup", "270000", _PANDAS_HINT) if s not in demo.stdout]
        if missing:
            print(f"mloda demo output lacks {missing}\nstdout:\n{demo.stdout}\nstderr:\n{demo.stderr}")
            return 1

        first_code = blocks[0].code
        pandas_code = first_code.replace('"PythonDictFramework"', '"PandasDataFrame"')
        assert pandas_code != first_code, 'README first example must name "PythonDictFramework"'
        pandas_script = root / "pandas_example.py"
        pandas_script.write_text(pandas_code, encoding="utf-8")
        # Not recorded in the time-to-first-success timings.
        # Safe: argv is the temp venv python and a script written above, no shell.
        pandas_run = subprocess.run(  # nosec B603
            [python, "-I", str(pandas_script)], capture_output=True, text=True, env=_clean_env(), timeout=300
        )
        if pandas_run.returncode == 0:
            print("pandas request must fail on a bare install, but exited 0")
            return 1
        if _PANDAS_HINT not in pandas_run.stderr or "No feature groups found" in pandas_run.stderr:
            print(f"pandas request must name the missing extra {_PANDAS_HINT}, stderr was:\n{pandas_run.stderr}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
