"""Run the README first example on a bare wheel install, as a new user would.

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

_FIRST_EXAMPLE = re.compile(
    rf"```{RUNNABLE_TAG}\n(.*?)```(?:\s*```{OUTPUT_TAG}\n(.*?)```)?",
    re.DOTALL,
)
_ABSENT_MODULES = ("pandas", "pyarrow", "numpy")
_PANDAS_HINT = 'pip install "mloda[pandas]"'
_timings: list[tuple[str, float]] = []


def first_example(readme_text: str) -> tuple[str, str | None]:
    """The first python block and the text block directly after it (None if absent)."""
    match = _FIRST_EXAMPLE.search(readme_text)
    assert match, f"README has no ```{RUNNABLE_TAG} block"
    return match.group(1), match.group(2)


def _run(label: str, argv: list[str], env: dict[str, str] | None = None) -> "subprocess.CompletedProcess[str]":
    start = time.monotonic()
    # Safe: argv is built from sys.executable, a temp venv and caller-given paths, no shell.
    result = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=300)  # nosec B603
    _timings.append((label, time.monotonic() - start))
    return result


def _clean_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if not k.startswith("MLODA_")}


def _summary() -> str:
    lines = [f"{seconds:7.2f}s  {label}" for label, seconds in _timings]
    lines.append(f"{sum(s for _, s in _timings):7.2f}s  total time to first success")
    return "\n".join(lines)


def main(readme: Path, dist: Path) -> int:
    code, expected = first_example(readme.read_text(encoding="utf-8"))
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

        script = Path(tmp) / "first_example.py"
        script.write_text(code, encoding="utf-8")
        run = _run("run README first example", [python, "-I", str(script)], env=_clean_env())
        print(_summary())
        if run.returncode != 0:
            print(f"first example failed, exit code {run.returncode}\nstdout:\n{run.stdout}\nstderr:\n{run.stderr}")
            return 1
        if run.stderr.strip():
            print(f"first example must run quietly, stderr was:\n{run.stderr}")
            return 1
        if expected is None:
            print("README has no ```text block after the first example")
            return 1
        if run.stdout.strip() != expected.strip():
            print(f"stdout differs from README\nexpected:\n{expected.strip()}\nactual:\n{run.stdout.strip()}")
            return 1

        pandas_code = code.replace('"PythonDictFramework"', '"PandasDataFrame"')
        assert pandas_code != code, 'README first example must name "PythonDictFramework"'
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
