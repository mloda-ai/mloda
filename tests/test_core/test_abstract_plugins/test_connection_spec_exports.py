"""ConnectionSpec is part of the public authoring surface: reachable from both mloda.user and
mloda.provider, and listed in each module's __all__.
"""

import mloda.provider as provider
import mloda.user as user


class TestConnectionSpecIsExportedFromUser:
    def test_importable(self) -> None:
        from mloda.user import ConnectionSpec

        assert ConnectionSpec is not None

    def test_listed_in_all(self) -> None:
        assert "ConnectionSpec" in user.__all__


class TestConnectionSpecIsExportedFromProvider:
    def test_importable(self) -> None:
        from mloda.provider import ConnectionSpec

        assert ConnectionSpec is not None

    def test_listed_in_all(self) -> None:
        assert "ConnectionSpec" in provider.__all__


class TestConnectionSpecIsTheSameObjectAcrossNamespaces:
    def test_user_and_provider_reexport_the_same_class(self) -> None:
        from mloda.provider import ConnectionSpec as ProviderConnectionSpec
        from mloda.user import ConnectionSpec as UserConnectionSpec

        assert UserConnectionSpec is ProviderConnectionSpec
