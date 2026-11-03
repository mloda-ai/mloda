"""Run every README python block on a bare wheel install, as a new user would.

Usage: python -m tests.readme_first_run <README.md> <dist-dir>
"""

import os
import re
import subprocess  # nosec B404
import sys
import tempfile
import time
from pathlib import Path

from tests.docs_corpus import OUTPUT_TAG, RUNNABLE_TAG

_PYTHON_BLOCK = re.compile(
    rf"```{RUNNABLE_TAG}\n(.*?)```(?:\s*```{OUTPUT_TAG}\n(.*?)```)?",
    re.DOTALL,
)
_ABSENT_MODULES = ("pandas", "pyarrow", "numpy")
_PANDAS_HINT = 'pip install "mloda[pandas]"'
_timings: list[tuple[str, float]] = []


def python_blocks(readme_text: str) -> list[tuple[str, str | None]]:
    """Every python block paired with the text block directly after it (None if absent), in document order."""
    return [(match.group(1), match.group(2)) for match in _PYTHON_BLOCK.finditer(readme_text)]


def first_example(readme_text: str) -> tuple[str, str | None]:
    """The first python block and the text block directly after it (None if absent)."""
    blocks = python_blocks(readme_text)
    assert blocks, f"README has no ```{RUNNABLE_TAG} block"
    return blocks[0]


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


def main(readme: Path, dist: Path) -> int:
    blocks = python_blocks(readme.read_text(encoding="utf-8"))
    assert blocks, f"README has no ```{RUNNABLE_TAG} block"
    wheels = sorted(dist.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel in {dist}, found {wheels}"
    with tempfile.TemporaryDirectory() as tmp:
        venv = Path(tmp) / "venv"
        python = str(venv / ("Scripts" if os.name == "nt" else "bin") / "python")
        # uv, not `python -m venv`: distro Pythons often lack ensurepip.
        created = _run("uv venv", ["uv", "venv", "--python", sys.executable, str(venv)])
        assert created.returncode == 0, f"{created.stdout}\n{created.stderr}"
        install = _run(
            "uv pip install mloda (offline wheel, no extras)",
            ["uv", "pip", "install", "--python", python, "--no-index", "--no-config", str(wheels[0])],
        )
        assert install.returncode == 0, f"{install.stdout}\n{install.stderr}"
        probe = "import importlib.util as u; print([m for m in %r if u.find_spec(m)])" % (_ABSENT_MODULES,)
        # Safe: argv is the temp venv python and a constant script, no shell.
        probed = subprocess.run(  # nosec B603
            [python, "-I", "-c", probe], capture_output=True, text=True, env=_clean_env(), timeout=60
        )
        present = probed.stdout.strip()
        assert present == "[]", f"bare install must not contain optional backends, found {present}"

        for index, (code, expected) in enumerate(blocks, 1):
            script = Path(tmp) / f"block_{index}.py"
            script.write_text(code, encoding="utf-8")
            # Only the first block counts towards time to first success.
            label = "run README first example" if index == 1 else None
            run = _run(label, [python, "-I", str(script)], env=_clean_env())
            if index == 1:
                print(_summary())
            failure = _block_failure(index, code, expected, run)
            if failure:
                print(failure)
                return 1
            if index == 1 and expected is None:
                print("README has no ```text block after the first example")
                return 1

        mloda_script = str(
            venv / ("Scripts" if os.name == "nt" else "bin") / ("mloda.exe" if os.name == "nt" else "mloda")
        )
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

        first_code = blocks[0][0]
        pandas_code = first_code.replace('"PythonDictFramework"', '"PandasDataFrame"')
        assert pandas_code != first_code, 'README first example must name "PythonDictFramework"'
        pandas_script = Path(tmp) / "pandas_example.py"
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
