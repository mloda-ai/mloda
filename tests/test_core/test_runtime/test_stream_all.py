"""Unit tests for mlodaAPI.stream_all classmethod."""

import inspect
import types


class TestStreamAllClassMethod:
    """Tests that stream_all is a @classmethod on mlodaAPI."""

    def test_stream_all_is_classmethod_on_mloda_api(self) -> None:
        from mloda.core.api.request import mlodaAPI

        assert isinstance(inspect.getattr_static(mlodaAPI, "stream_all"), classmethod)

    def test_stream_all_returns_generator(self) -> None:
        from mloda.core.api.request import mlodaAPI
        from mloda.user import Feature, Features, ParallelizationMode
        from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

        features = Features([Feature(name="EngineRunnerTest1", initial_requested_data=True)])
        result = mlodaAPI.stream_all(
            features,
            compute_frameworks={PyArrowTable},
            parallelization_modes={ParallelizationMode.SYNC},
        )

        assert isinstance(result, types.GeneratorType)

    def test_stream_all_accessible_via_mloda_alias(self) -> None:
        from mloda.user import mloda

        assert callable(mloda.stream_all)


class TestStreamAllImportFromUser:
    """Tests that stream_all is importable from mloda.user."""

    def test_stream_all_importable_from_user_module(self) -> None:
        from mloda.user import stream_all

        assert callable(stream_all)

    def test_stream_all_returns_generator_via_user_import(self) -> None:
        from mloda.user import Feature, Features, ParallelizationMode, stream_all
        from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

        features = Features([Feature(name="EngineRunnerTest1", initial_requested_data=True)])
        result = stream_all(
            features,
            compute_frameworks={PyArrowTable},
            parallelization_modes={ParallelizationMode.SYNC},
        )

        assert isinstance(result, types.GeneratorType)
        assert not isinstance(result, list)
