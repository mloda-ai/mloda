"""mloda.provider must re-export the resolution-debug surface, identical to mloda.user (issue #855)."""

import pytest

import mloda.provider as provider
import mloda.steward as steward
import mloda.user as user

RESOLUTION_SURFACE = ["resolve_feature", "FeatureResolutionError", "ResolutionRecord", "ResolutionDiagnosis"]


class TestResolutionSurfaceIsExported:
    @pytest.mark.parametrize("name", RESOLUTION_SURFACE)
    def test_name_is_public(self, name: str) -> None:
        assert hasattr(provider, name)
        assert name in provider.__all__, f"{name} should be listed in mloda.provider.__all__"


class TestResolutionSurfaceMatchesUser:
    @pytest.mark.parametrize("name", RESOLUTION_SURFACE)
    def test_reexport_is_the_same_object(self, name: str) -> None:
        """A divergent duplicate import would break debugging tools that compare identity."""
        assert getattr(provider, name, None) is getattr(user, name), (
            f"mloda.provider.{name} must be the same object as mloda.user.{name}"
        )


class TestResolutionSurfaceMatchesSteward:
    @pytest.mark.parametrize("name", RESOLUTION_SURFACE)
    def test_reexport_is_the_same_object(self, name: str) -> None:
        """Issue #855 pins the surface against mloda.steward too, not just mloda.user."""
        assert getattr(provider, name, None) is getattr(steward, name), (
            f"mloda.provider.{name} must be the same object as mloda.steward.{name}"
        )
