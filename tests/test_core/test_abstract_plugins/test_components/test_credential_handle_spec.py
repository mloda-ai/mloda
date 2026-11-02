"""credential_handle_spec: plan-time handle validation, plan records and display."""

from abc import abstractmethod
from collections.abc import Callable
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.domain import Domain
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.api.plan_lock import _lock_text
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from tests.helpers.credential_handle_stubs import HANDLE_KEY, ChsRawSecretFG, ChsSource, ChsVaultFG
from mloda.provider import (
    ComputeFramework,
    FeatureGroup,
    credential_handle_spec,
)
from mloda.steward import plan_content_hash
from mloda.user import Credential, Feature, FeatureName, FilterType, GlobalFilter, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

SECRET = "chs-secret-value-1234"  # nosec B105
OTHER_SECRET = "chs-other-secret-5678"  # nosec B105
BYTES_SECRET = b"chs-bytes-secret-9012"


class ChsFallbackFG(ChsSource):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_vault_a"}


class ChsDefaultFG(ChsSource):
    PROPERTY_MAPPING = {
        "chs_default_handle": credential_handle_spec("Defaulted handle.", default="vault-default"),
        "chs_optional_handle": credential_handle_spec("Optional handle.", default=None),
    }

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_default_a"}


class ChsTwoHandleFG(ChsSource):
    PROPERTY_MAPPING = {
        "chs_two_z_handle": credential_handle_spec("Second handle."),
        "chs_two_a_handle": credential_handle_spec("First handle."),
    }

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_two_a"}


class ChsFieldsFG(ChsSource):
    PROPERTY_MAPPING = {
        "chs_fields_handle": credential_handle_spec("Needs fields.", required_fields=("chs_user", "chs_pass"))
    }

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_fields_a"}


class ChsCustomMatchFG(ChsSource):
    PROPERTY_MAPPING = {"chs_custom_handle": credential_handle_spec("Custom matcher handle.")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_custom_a"}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == "chs_custom_a"


class ChsAbstractFG(ChsSource):
    PROPERTY_MAPPING = {"chs_abstract_handle": credential_handle_spec("Abstract handle.")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_abstract_a"}

    @abstractmethod
    def chs_hook(self) -> None: ...


class ChsConcreteFG(ChsSource):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_abstract_a"}


class ChsGatedFG(ChsSource):
    PROPERTY_MAPPING = {"chs_gated_handle": credential_handle_spec("Gated handle.")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_domain_a"}

    @classmethod
    def get_domain(cls) -> Domain:
        return Domain("chs_dom_a")


class ChsOpenDomainFG(ChsSource):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_domain_a"}

    @classmethod
    def get_domain(cls) -> Domain:
        return Domain("chs_dom_b")


class ChsFrameworkLoserFG(ChsSource):
    PROPERTY_MAPPING = {"chs_loser_handle": credential_handle_spec("Loses on framework.")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_loser_a"}

    @classmethod
    def supports_compute_framework(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        compute_framework: type[ComputeFramework],
        *,
        part: str | None = None,
    ) -> bool:
        return False


class ChsLoserWinnerFG(ChsSource):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_loser_a"}


def _plugins(*groups: type[FeatureGroup]) -> PluginCollector:
    return PluginCollector.enabled_feature_groups(set(groups))


def _dac(**handles: Any) -> DataAccessCollection:
    return DataAccessCollection(credentials={handle: value for handle, value in handles.items()})


def _vault(handle: str, name: str = "chs_vault_a") -> Feature:
    return Feature(name, options={HANDLE_KEY: handle})


def _prepare(features: list[Feature], dac: DataAccessCollection | None, plugins: PluginCollector) -> Any:
    return mloda.prepare(
        features, compute_frameworks=[PandasDataFrame], data_access_collection=dac, plugin_collector=plugins
    )


def _explain(features: list[Feature], dac: DataAccessCollection | None, plugins: PluginCollector) -> Any:
    return mloda.explain(
        features, compute_frameworks=[PandasDataFrame], data_access_collection=dac, plugin_collector=plugins
    )


def _run_all(features: list[Feature], dac: DataAccessCollection | None, plugins: PluginCollector) -> Any:
    return mloda.run_all(
        features, compute_frameworks=[PandasDataFrame], data_access_collection=dac, plugin_collector=plugins
    )


ENTRY_POINTS = [_prepare, _explain, _run_all]
Entry = Callable[[list[Feature], DataAccessCollection | None, PluginCollector], Any]


def _refusal(
    entry: Entry, features: list[Feature], dac: DataAccessCollection | None, *groups: type[FeatureGroup]
) -> str:
    with pytest.raises(ValueError) as exc:
        entry(features, dac, _plugins(*groups))
    return str(exc.value)


def _steps(session: Any, group: type[FeatureGroup]) -> list[FeatureGroupStep]:
    return [s for s in session.engine.execution_planner if isinstance(s, FeatureGroupStep) and s.feature_group is group]


class TestPlanTimeRefusals:
    @pytest.mark.parametrize("entry", ENTRY_POINTS)
    def test_unknown_handle_names_the_handle_and_the_available_ones(self, entry: Entry) -> None:
        dac = _dac(**{"vault-main": {"key": SECRET}, "vault-extra": {"key": OTHER_SECRET}})

        message = _refusal(entry, [_vault("vault-typo")], dac, ChsVaultFG)

        assert "vault-typo" in message
        assert "vault-main" in message
        assert "vault-extra" in message
        assert HANDLE_KEY in message
        assert SECRET not in message
        assert OTHER_SECRET not in message

    @pytest.mark.parametrize("entry", ENTRY_POINTS)
    def test_unknown_handle_without_a_collection_names_the_handle(self, entry: Entry) -> None:
        message = _refusal(entry, [_vault("vault-typo")], None, ChsVaultFG)

        assert "vault-typo" in message

    @pytest.mark.parametrize("entry", ENTRY_POINTS)
    def test_a_typo_ends_planning_even_when_another_group_matches_the_name(self, entry: Entry) -> None:
        message = _refusal(
            entry, [_vault("vault-typo")], _dac(**{"vault-main": {"key": SECRET}}), ChsVaultFG, ChsFallbackFG
        )

        assert "vault-typo" in message

    def test_a_handle_registered_under_another_kind_names_both_kinds(self) -> None:
        dac = DataAccessCollection(files={"vault-main": "chs-file.csv"})

        message = _refusal(_prepare, [_vault("vault-main")], dac, ChsVaultFG)

        assert "vault-main" in message
        assert "file" in message
        assert "credentials" in message

    def test_auto_named_credentials_are_counted_not_named(self) -> None:
        dac = DataAccessCollection(credentials=Credential(host="chs-auto-host-1"))

        message = _refusal(_prepare, [_vault("vault-main")], dac, ChsVaultFG)

        assert "unnamed" in message
        assert "_auto_credentials_0" not in message
        assert "chs-auto-host-1" not in message

    def test_an_auto_handle_cannot_be_declared(self) -> None:
        dac = DataAccessCollection(credentials=Credential(host="chs-auto-host-1"))

        message = _refusal(_prepare, [_vault("_auto_credentials_0")], dac, ChsVaultFG)

        assert "unnamed" in message
        assert "chs-auto-host-1" not in message

    @pytest.mark.parametrize("value", [987654321, ["vault-main"], ("vault-main",), ""])
    def test_a_non_str_or_empty_handle_aborts_naming_the_key(self, value: Any) -> None:
        feature = Feature("chs_vault_a", options={HANDLE_KEY: value})

        message = _refusal(_prepare, [feature], _dac(**{"vault-main": {"key": SECRET}}), ChsVaultFG, ChsFallbackFG)

        assert HANDLE_KEY in message
        assert "987654321" not in message

    def test_a_missing_required_field_names_the_handle_and_the_field(self) -> None:
        dac = _dac(**{"vault-main": {"chs_user": SECRET}})
        feature = Feature("chs_fields_a", options={"chs_fields_handle": "vault-main"})

        message = _refusal(_prepare, [feature], dac, ChsFieldsFG)

        assert "vault-main" in message
        assert "chs_pass" in message
        assert SECRET not in message

    def test_a_credential_with_every_required_field_plans(self) -> None:
        dac = _dac(**{"vault-main": {"chs_user": SECRET, "chs_pass": OTHER_SECRET}})
        feature = Feature("chs_fields_a", options={"chs_fields_handle": "vault-main"})

        assert mloda.explain(
            [feature],
            compute_frameworks=[PandasDataFrame],
            data_access_collection=dac,
            plugin_collector=_plugins(ChsFieldsFG),
        )

    @pytest.mark.parametrize(
        ("credential", "path_part"),
        [
            ({"chs_blob": BYTES_SECRET}, "chs_blob"),
            ({"chs_outer": {"chs_inner": [BYTES_SECRET]}}, "chs_outer"),
            ({"chs_outer": ({"chs_inner": bytearray(BYTES_SECRET)},)}, "chs_outer"),
            ({"host": {"password": BYTES_SECRET}}, "host"),
            ({"password": {"host": BYTES_SECRET}}, "password"),
            ({"chs_outer": Credential(chs_inner=BYTES_SECRET)}, "chs_outer"),
        ],
    )
    def test_a_bytes_leaf_is_refused_naming_the_handle_and_key_path(
        self, credential: dict[str, Any], path_part: str
    ) -> None:
        message = _refusal(_prepare, [_vault("vault-main")], _dac(**{"vault-main": credential}), ChsVaultFG)

        assert "vault-main" in message
        assert path_part in message
        assert "hex" in message or "base64" in message
        assert BYTES_SECRET.decode() not in message

    def test_a_cyclic_credential_does_not_hang_the_bytes_walk(self) -> None:
        cyclic: dict[str, Any] = {"chs_field": "chs-cyclic-value-1"}
        cyclic["chs_self"] = cyclic

        mloda.explain(
            [_vault("vault-main")],
            compute_frameworks=[PandasDataFrame],
            data_access_collection=_dac(**{"vault-main": cyclic}),
            plugin_collector=_plugins(ChsVaultFG),
        )

    def test_bytes_in_another_registered_credential_stay_accepted_and_resolve(self) -> None:
        dac = _dac(**{"vault-main": {"key": SECRET}, "source-db": {"dsn": BYTES_SECRET}})

        results = mloda.run_all(
            [_vault("vault-main")],
            compute_frameworks=[PandasDataFrame],
            data_access_collection=dac,
            plugin_collector=_plugins(ChsVaultFG),
        )

        assert results[0]["chs_vault_a"].tolist() == [1, 2, 3]

    def test_bytes_under_a_safe_key_are_accepted(self) -> None:
        dac = _dac(**{"vault-main": {"key": SECRET, "path": BYTES_SECRET}})

        results = mloda.run_all(
            [_vault("vault-main")],
            compute_frameworks=[PandasDataFrame],
            data_access_collection=dac,
            plugin_collector=_plugins(ChsVaultFG),
        )

        assert results[0]["chs_vault_a"].tolist() == [1, 2, 3]

    @pytest.mark.parametrize("entry", ENTRY_POINTS)
    def test_a_handle_in_context_is_refused_naming_the_key_and_the_group_placement(self, entry: Entry) -> None:
        feature = Feature("chs_vault_a", options=Options(group={}, context={HANDLE_KEY: "vault-main"}))

        message = _refusal(entry, [feature], _dac(**{"vault-main": {"key": SECRET}}), ChsVaultFG)

        assert HANDLE_KEY in message
        assert "group" in message
        assert SECRET not in message

    def test_a_custom_matcher_override_is_still_validated(self) -> None:
        feature = Feature("chs_custom_a", options={"chs_custom_handle": "vault-typo"})

        message = _refusal(_prepare, [feature], _dac(**{"vault-main": {"key": SECRET}}), ChsCustomMatchFG)

        assert "vault-typo" in message

    def test_an_abstract_candidate_is_not_validated(self) -> None:
        feature = Feature("chs_abstract_a", options={"chs_abstract_handle": "vault-typo"})

        session = _prepare([feature], None, _plugins(ChsAbstractFG, ChsConcreteFG))

        assert _steps(session, ChsConcreteFG)

    def test_a_domain_gated_candidate_with_a_bad_handle_does_not_abort(self) -> None:
        feature = Feature("chs_domain_a", options={"chs_gated_handle": "vault-typo"}, domain="chs_dom_b")

        session = _prepare([feature], None, _plugins(ChsGatedFG, ChsOpenDomainFG))

        assert _steps(session, ChsOpenDomainFG)

    def test_a_candidate_that_later_loses_on_a_framework_still_refuses(self) -> None:
        feature = Feature("chs_loser_a", options={"chs_loser_handle": "vault-typo"})

        message = _refusal(_prepare, [feature], None, ChsFrameworkLoserFG, ChsLoserWinnerFG)

        assert "vault-typo" in message

    def test_a_refused_default_names_the_default_handle(self) -> None:
        message = _refusal(_prepare, [Feature("chs_default_a")], _dac(**{"vault-main": {"key": SECRET}}), ChsDefaultFG)

        assert "vault-default" in message
        assert SECRET not in message


class TestGlobalFilterFeatures:
    """A feature added only through a global filter is validated like any requested feature."""

    @staticmethod
    def _prepare_with_filter(filter_feature: Feature) -> Any:
        global_filter = GlobalFilter()
        global_filter.add_filter(filter_feature, FilterType.EQUAL, {"value": 1})
        return mloda.prepare(
            [_vault("vault-main")],
            compute_frameworks=[PandasDataFrame],
            data_access_collection=_dac(**{"vault-main": {"key": SECRET}}),
            plugin_collector=_plugins(ChsVaultFG),
            global_filter=global_filter,
        )

    def test_an_unknown_handle_is_refused_naming_the_handle(self) -> None:
        with pytest.raises(ValueError) as exc:
            self._prepare_with_filter(_vault("vault-typo", "chs_vault_b"))

        assert "vault-typo" in str(exc.value)
        assert SECRET not in str(exc.value)

    def test_a_non_str_handle_value_is_refused_without_printing_it(self) -> None:
        feature = Feature("chs_vault_b", options={HANDLE_KEY: {"password": OTHER_SECRET}})

        with pytest.raises(ValueError) as exc:
            self._prepare_with_filter(feature)

        assert OTHER_SECRET not in str(exc.value)

    def test_a_handle_in_context_is_refused_naming_the_key(self) -> None:
        feature = Feature("chs_vault_b", options=Options(group={}, context={HANDLE_KEY: "vault-main"}))

        with pytest.raises(ValueError) as exc:
            self._prepare_with_filter(feature)

        assert HANDLE_KEY in str(exc.value)
        assert "group" in str(exc.value)


class TestPlanRecords:
    def test_two_handles_split_two_steps_each_recording_its_handle(self) -> None:
        dac = _dac(**{"vault-a": {"key": SECRET}, "vault-b": {"key": OTHER_SECRET}})

        session = _prepare(
            [_vault("vault-b", "chs_vault_a"), _vault("vault-a", "chs_vault_b")], dac, _plugins(ChsVaultFG)
        )

        steps = _steps(session, ChsVaultFG)
        assert sorted(s.credential_handles for s in steps) == [("vault-a",), ("vault-b",)]

    def test_recorded_handles_are_sorted_by_handle_name(self) -> None:
        dac = _dac(**{"vault-a": {"key": SECRET}, "vault-b": {"key": OTHER_SECRET}})
        feature = Feature("chs_two_a", options={"chs_two_a_handle": "vault-b", "chs_two_z_handle": "vault-a"})

        session = _prepare([feature], dac, _plugins(ChsTwoHandleFG))

        assert [s.credential_handles for s in _steps(session, ChsTwoHandleFG)] == [("vault-a", "vault-b")]

    def test_a_str_default_is_recorded_and_shown_and_none_declares_nothing(self) -> None:
        dac = _dac(**{"vault-default": {"key": SECRET}})

        session = _prepare([Feature("chs_default_a")], dac, _plugins(ChsDefaultFG))
        plan = _explain([Feature("chs_default_a")], dac, _plugins(ChsDefaultFG))

        assert [s.credential_handles for s in _steps(session, ChsDefaultFG)] == [("vault-default",)]
        snapshot = next(s for s in plan if s.feature_group is ChsDefaultFG)
        assert snapshot.feature_set_options is not None
        assert snapshot.feature_set_options.group["chs_default_handle"] == "vault-default"

    def test_a_group_without_a_declaration_records_no_handles(self) -> None:
        session = _prepare([Feature("chs_vault_a")], None, _plugins(ChsFallbackFG))

        assert [s.credential_handles for s in _steps(session, ChsFallbackFG)] == [()]


class TestDisplay:
    def _session(self, handle: str = "vault-main") -> Any:
        return _prepare([_vault(handle)], _dac(**{handle: {"key": SECRET}}), _plugins(ChsVaultFG))

    def test_str_options_shows_the_handle_under_a_credential_shaped_key(self) -> None:
        session = self._session()

        options = next(iter(session.engine.feature_group_collection[ChsVaultFG])).options

        assert "vault-main" in str(options)
        assert SECRET not in str(options)

    def test_the_snapshot_shows_the_handle_and_no_credential_value(self) -> None:
        plan = mloda.explain(
            [_vault("vault-main")],
            compute_frameworks=[PandasDataFrame],
            data_access_collection=_dac(**{"vault-main": {"key": SECRET}}),
            plugin_collector=_plugins(ChsVaultFG),
        )

        step = next(s for s in plan if s.feature_group is ChsVaultFG)
        assert step.feature_set_options is not None
        assert step.feature_set_options.group[HANDLE_KEY] == "vault-main"
        for text in (str(step.feature_set_options), repr(step), str(plan)):
            assert SECRET not in text

    def test_the_content_hash_changes_with_the_handle_name_and_the_lock_text_does_not(self) -> None:
        def explain(handle: str) -> Any:
            return mloda.explain(
                [_vault(handle)],
                compute_frameworks=[PandasDataFrame],
                data_access_collection=_dac(**{handle: {"key": SECRET}}),
                plugin_collector=_plugins(ChsVaultFG),
            )

        plan_a, plan_b = explain("vault-a"), explain("vault-b")

        assert plan_content_hash(plan_a) != plan_content_hash(plan_b)
        assert _lock_text(plan_a) == _lock_text(plan_b)
        assert SECRET not in _lock_text(plan_a)

    def test_a_raw_secret_under_the_same_key_in_another_group_stays_masked(self) -> None:
        def explain(secret: str) -> Any:
            return mloda.explain(
                [Feature("chs_raw_a", options={HANDLE_KEY: secret})],
                compute_frameworks=[PandasDataFrame],
                plugin_collector=_plugins(ChsRawSecretFG),
            )

        plan_a, plan_b = explain(SECRET), explain(OTHER_SECRET)

        step = next(s for s in plan_a if s.feature_group is ChsRawSecretFG)
        assert step.feature_set_options is not None
        assert step.feature_set_options.group[HANDLE_KEY] == "***"
        assert plan_content_hash(plan_a) == plan_content_hash(plan_b)
        assert SECRET not in str(step.feature_set_options)

    def test_a_raw_secret_in_another_group_is_masked_in_str_options_after_intake(self) -> None:
        session = _prepare([Feature("chs_raw_a", options={HANDLE_KEY: SECRET})], None, _plugins(ChsRawSecretFG))

        options = next(iter(session.engine.feature_group_collection[ChsRawSecretFG])).options

        assert SECRET not in str(options)

    def test_a_mark_copied_from_another_group_does_not_survive_intake(self) -> None:
        options = Options(group={HANDLE_KEY: SECRET})
        options.credential_handle_keys = frozenset({HANDLE_KEY})

        session = _prepare([Feature("chs_raw_a", options=options)], None, _plugins(ChsRawSecretFG))

        stored = next(iter(session.engine.feature_group_collection[ChsRawSecretFG])).options
        assert SECRET not in str(stored)
        assert stored.credential_handle_keys == frozenset()
