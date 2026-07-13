"""Registers the EngineRunnerTest feature groups so test_stream_all.py resolves them standalone.

stream_all now plans eagerly, so ``EngineRunnerTest1`` must be a loaded FeatureGroup subclass at
call time, not only when the full suite's collection has imported its defining module.
"""

import tests.test_core.test_integration.test_core.test_runner_one_compute_framework  # noqa: F401
