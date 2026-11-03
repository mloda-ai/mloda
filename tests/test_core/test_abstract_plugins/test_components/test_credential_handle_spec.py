"""credential_handle_spec: plan-time handle validation, plan records and display."""

import re
from abc import abstractmethod
from collections.abc import Callable
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.domain import Domain
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.api.plan_lock import _lock_text
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from tests.helpers.credential_handle_stubs import HANDLE_KEY, ChsRawSecretFG, ChsSource, ChsVaultFG
from mloda.provider import (
    ComputeFramework,
    FeatureGroup,
    PropertySpec,
    credential_handle_spec,
    property_spec,
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
        return {"chs_vault_a", "chs_plain_a", "chs_both_a", "chs_custom_a", "chs_up_req"}


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


class ChsPlainRequiredFG(ChsSource):
    PROPERTY_MAPPING = {"chs_plain_req": property_spec("Required plain option.")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_plain_a"}


class ChsBothMissingFG(ChsSource):
    PROPERTY_MAPPING = {
        "chs_both_plain": property_spec("Required plain option."),
        "chs_both_handle": credential_handle_spec("Required handle."),
    }

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_both_a"}


class ChsPairFirstFG(ChsSource):
    PROPERTY_MAPPING = {"chs_pair_first_handle": credential_handle_spec("First handle.")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_pair_a"}


class ChsPairSecondFG(ChsSource):
    PROPERTY_MAPPING = {"chs_pair_second_handle": credential_handle_spec("Second handle.")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_pair_a"}


class ChsNeverMatchFG(ChsCustomMatchFG):
    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return False


class ChsNoneDefaultFG(ChsSource):
    PROPERTY_MAPPING = {"chs_none_handle": credential_handle_spec("Optional.", default=None)}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_none_a"}


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
            ({"host": {"password": BYTES_SECRET}}, "host.password"),
            ({"password": {"host": BYTES_SECRET}}, "password.host"),
            ({"chs_outer": Credential(chs_inner=BYTES_SECRET)}, "chs_outer.chs_inner"),
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


class TestMissingHandleKeyAborts:
    """A required handle key that is absent aborts planning instead of falling back to another group."""

    @pytest.mark.parametrize("entry", ENTRY_POINTS)
    def test_a_missing_handle_key_aborts_despite_a_fallback_group(self, entry: Entry) -> None:
        message = _refusal(entry, [Feature("chs_vault_a")], None, ChsVaultFG, ChsFallbackFG)

        assert HANDLE_KEY in message
        assert "ChsVaultFG" in message

    @pytest.mark.parametrize("with_fallback", [True, False])
    def test_a_misspelled_handle_key_names_the_close_match(self, with_fallback: bool) -> None:
        typo = "chs_vault_tokne"
        feature = Feature("chs_vault_a", options={typo: "vault-main"})
        groups: tuple[type[FeatureGroup], ...] = (ChsVaultFG, ChsFallbackFG) if with_fallback else (ChsVaultFG,)

        message = _refusal(_prepare, [feature], _dac(**{"vault-main": {"key": SECRET}}), *groups)

        assert HANDLE_KEY in message
        assert "ChsVaultFG" in message
        assert f"did you mean '{typo}'" in message
        assert SECRET not in message

    def test_a_missing_handle_key_without_a_close_match_has_no_suggestion(self) -> None:
        feature = Feature("chs_vault_a", options={"chs_unrelated": "x"})

        message = _refusal(_prepare, [feature], None, ChsVaultFG, ChsFallbackFG)

        assert HANDLE_KEY in message
        assert "did you mean" not in message

    def test_a_group_missing_a_handle_and_a_plain_key_names_the_handle_key(self) -> None:
        message = _refusal(_prepare, [Feature("chs_both_a")], None, ChsBothMissingFG, ChsFallbackFG)

        assert "chs_both_handle" in message

    def test_another_handle_declaring_group_that_matches_wins_over_the_abort(self) -> None:
        feature = Feature("chs_pair_a", options={"chs_pair_second_handle": "vault-main"})

        session = _prepare(
            [feature], _dac(**{"vault-main": {"key": SECRET}}), _plugins(ChsPairFirstFG, ChsPairSecondFG)
        )

        assert _steps(session, ChsPairSecondFG)
        assert not _steps(session, ChsPairFirstFG)

    def test_a_group_missing_two_handle_keys_names_both(self) -> None:
        message = _refusal(_prepare, [Feature("chs_two_a")], None, ChsTwoHandleFG)

        assert "chs_two_a_handle" in message
        assert "chs_two_z_handle" in message

    def test_a_missing_plain_required_key_keeps_the_fallback(self) -> None:
        session = _prepare([Feature("chs_plain_a")], None, _plugins(ChsPlainRequiredFG, ChsFallbackFG))

        assert _steps(session, ChsFallbackFG)

    def test_a_none_default_handle_without_a_key_still_resolves_to_the_declaring_group(self) -> None:
        session = _prepare([Feature("chs_none_a")], None, _plugins(ChsNoneDefaultFG))

        assert _steps(session, ChsNoneDefaultFG)

    def test_a_custom_matcher_that_returns_false_keeps_the_fallback(self) -> None:
        session = _prepare([Feature("chs_custom_a")], None, _plugins(ChsNeverMatchFG, ChsFallbackFG))

        assert _steps(session, ChsFallbackFG)


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

    def test_the_content_hash_and_the_lock_text_change_with_the_handle_name(self) -> None:
        def explain(handle: str) -> Any:
            return mloda.explain(
                [_vault(handle)],
                compute_frameworks=[PandasDataFrame],
                data_access_collection=_dac(**{handle: {"key": SECRET}}),
                plugin_collector=_plugins(ChsVaultFG),
            )

        plan_a, plan_b = explain("vault-a"), explain("vault-b")

        assert plan_content_hash(plan_a) != plan_content_hash(plan_b)
        assert _lock_text(plan_a) != _lock_text(plan_b)
        assert "vault-a" in _lock_text(plan_a)
        assert _lock_text(plan_a).replace("vault-a", "vault-b") == _lock_text(plan_b)
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


class ChsConsumerFG(FeatureGroup):
    PROPERTY_MAPPING = {HANDLE_KEY: credential_handle_spec("Consumer handle.")}
    INPUT_NAME = "chs_src_a"
    FORWARD: frozenset[str] | None = None

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_consumer_a"}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature(self.INPUT_NAME, forward_group=self.FORWARD)}

    @classmethod
    def calculate_feature(cls, data: Any, features: Any) -> Any:
        return data


class ChsConsumerTwoFG(ChsConsumerFG):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_consumer_b"}


class ChsConsumerOptUpFG(ChsConsumerFG):
    INPUT_NAME = "chs_up_opt"

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_consumer_opt"}


class ChsConsumerAllowFG(ChsConsumerFG):
    INPUT_NAME = "chs_up_opt"
    FORWARD = frozenset({HANDLE_KEY})

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_consumer_allow"}


class ChsConsumerReqUpFG(ChsConsumerFG):
    INPUT_NAME = "chs_up_req"

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_consumer_req"}


class ChsConsumerStrDefaultFG(ChsConsumerFG):
    PROPERTY_MAPPING = {HANDLE_KEY: credential_handle_spec("Defaulted consumer handle.", default="vault-default")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_consumer_strdef"}


class ChsPlainSourceFG(ChsSource):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_src_a"}


class ChsOptionalUpstreamFG(ChsSource):
    PROPERTY_MAPPING = {HANDLE_KEY: credential_handle_spec("Optional upstream handle.", default=None)}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_up_opt"}


class ChsRequiredUpstreamFG(ChsSource):
    PROPERTY_MAPPING = {HANDLE_KEY: credential_handle_spec("Required upstream handle.")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_up_req"}


def _consumer(name: str, handle: str | None = "vault-a") -> Feature:
    return Feature(name, options={} if handle is None else {HANDLE_KEY: handle})


def _reader_group_keys(session: Any, group: type[FeatureGroup]) -> set[str]:
    return {key for feature in session.engine.feature_group_collection[group] for key in feature.options.group}


class TestHandleStaysOnTheConsumer:
    """A handle option is never forwarded to input features."""

    def test_the_reader_step_carries_no_handle_and_the_consumer_records_it(self) -> None:
        dac = _dac(**{"vault-a": {"key": SECRET}})
        plugins = _plugins(ChsConsumerFG, ChsPlainSourceFG)

        session = _prepare([_consumer("chs_consumer_a")], dac, plugins)
        plan = _explain([_consumer("chs_consumer_a")], dac, plugins)

        assert [s.credential_handles for s in _steps(session, ChsConsumerFG)] == [("vault-a",)]
        assert [s.credential_handles for s in _steps(session, ChsPlainSourceFG)] == [()]
        assert HANDLE_KEY not in _reader_group_keys(session, ChsPlainSourceFG)
        reader = next(s for s in plan if s.feature_group is ChsPlainSourceFG)
        assert reader.feature_set_options is not None
        assert HANDLE_KEY not in reader.feature_set_options.group
        assert "vault-a" not in repr(reader)

    def test_two_consumers_with_different_handles_share_one_reader_step(self) -> None:
        dac = _dac(**{"vault-a": {"key": SECRET}, "vault-b": {"key": OTHER_SECRET}})

        session = _prepare(
            [_consumer("chs_consumer_a", "vault-a"), _consumer("chs_consumer_b", "vault-b")],
            dac,
            _plugins(ChsConsumerFG, ChsConsumerTwoFG, ChsPlainSourceFG),
        )

        assert len(_steps(session, ChsPlainSourceFG)) == 1
        assert sorted(s.credential_handles for g in (ChsConsumerFG, ChsConsumerTwoFG) for s in _steps(session, g)) == [
            ("vault-a",),
            ("vault-b",),
        ]

    def test_an_optional_upstream_declaring_the_same_key_records_nothing(self) -> None:
        dac = _dac(**{"vault-a": {"key": SECRET}})

        session = _prepare([_consumer("chs_consumer_opt")], dac, _plugins(ChsConsumerOptUpFG, ChsOptionalUpstreamFG))

        assert [s.credential_handles for s in _steps(session, ChsOptionalUpstreamFG)] == [()]

    def test_a_missing_handle_key_on_an_upstream_input_aborts_despite_a_fallback_group(self) -> None:
        dac = _dac(**{"vault-a": {"key": SECRET}})
        plugins = _plugins(ChsConsumerReqUpFG, ChsRequiredUpstreamFG, ChsFallbackFG)

        with pytest.raises(ValueError, match=HANDLE_KEY):
            _prepare([_consumer("chs_consumer_req")], dac, plugins)

    def test_a_required_upstream_own_handle_is_not_received_from_the_consumer_and_aborts_naming_it(self) -> None:
        dac = _dac(**{"vault-a": {"key": SECRET}})

        with pytest.raises(
            ValueError,
            match=r"ChsRequiredUpstreamFG requires credential handle option 'chs_vault_token'",
        ):
            _prepare([_consumer("chs_consumer_req")], dac, _plugins(ChsConsumerReqUpFG, ChsRequiredUpstreamFG))

    def test_an_explicit_allowlist_on_the_input_feature_forwards_it(self) -> None:
        dac = _dac(**{"vault-a": {"key": SECRET}})

        session = _prepare([_consumer("chs_consumer_allow")], dac, _plugins(ChsConsumerAllowFG, ChsOptionalUpstreamFG))

        assert [s.credential_handles for s in _steps(session, ChsOptionalUpstreamFG)] == [("vault-a",)]

    def test_a_plain_option_with_forward_to_inputs_false_is_not_forwarded(self) -> None:
        class ChsConsumerPlainFG(ChsConsumerFG):
            PROPERTY_MAPPING = {
                "chs_local_opt": property_spec("Local option.", default=None, context=False, forward_to_inputs=False)
            }

            @classmethod
            def feature_names_supported(cls) -> set[str]:
                return {"chs_consumer_plain"}

        feature = Feature("chs_consumer_plain", options={"chs_local_opt": "local"})

        session = _prepare([feature], None, _plugins(ChsConsumerPlainFG, ChsPlainSourceFG))

        assert "chs_local_opt" not in _reader_group_keys(session, ChsPlainSourceFG)

    def test_a_str_default_on_the_consumer_handle_is_not_forwarded(self) -> None:
        dac = _dac(**{"vault-default": {"key": SECRET}})

        session = _prepare(
            [_consumer("chs_consumer_strdef", None)], dac, _plugins(ChsConsumerStrDefaultFG, ChsPlainSourceFG)
        )

        assert [s.credential_handles for s in _steps(session, ChsConsumerStrDefaultFG)] == [("vault-default",)]
        assert HANDLE_KEY not in _reader_group_keys(session, ChsPlainSourceFG)


_CAPTURE = f"(?P<{HANDLE_KEY}>\\w+)"
_PATTERN_FORMS = [
    pytest.param(lambda: f".*__{_CAPTURE}_chsb$", id="str"),
    pytest.param(lambda: re.compile(f".*__{_CAPTURE}_chsb$"), id="compiled"),
    pytest.param(lambda: [r"^chsb_(\d+)$", f".*__{_CAPTURE}_chsb$"], id="list"),
]
_HANDLE_MAPPING: dict[str, PropertySpec] = {HANDLE_KEY: credential_handle_spec("Handle.", default=None)}


class TestHandleIsNeverBoundFromTheName:
    """A named capture over a handle key is refused at class definition."""

    @pytest.mark.parametrize("make_pattern", _PATTERN_FORMS)
    def test_a_mixin_group_capturing_the_handle_key_is_refused(self, make_pattern: Callable[[], Any]) -> None:
        pattern = make_pattern()

        with pytest.raises(ValueError) as exc:

            class ChsBoundMixinGroupB1(FeatureChainParserMixin):
                PREFIX_PATTERN = pattern
                PROPERTY_MAPPING = _HANDLE_MAPPING

        assert "ChsBoundMixinGroupB1" in str(exc.value)
        assert HANDLE_KEY in str(exc.value)
        assert "feature name" in str(exc.value)

    @pytest.mark.parametrize("make_pattern", _PATTERN_FORMS)
    def test_a_plain_group_capturing_the_handle_key_in_a_suffix_pattern_is_refused(
        self, make_pattern: Callable[[], Any]
    ) -> None:
        pattern = make_pattern()

        with pytest.raises(ValueError) as exc:

            class ChsBoundPlainGroupB2(FeatureGroup):
                SUFFIX_PATTERN = pattern
                PROPERTY_MAPPING = _HANDLE_MAPPING

        assert "ChsBoundPlainGroupB2" in str(exc.value)
        assert HANDLE_KEY in str(exc.value)
        assert "feature name" in str(exc.value)

    def test_a_plain_group_capturing_the_handle_key_in_a_prefix_pattern_is_refused(self) -> None:
        with pytest.raises(ValueError) as exc:

            class ChsBoundPlainGroupB3(FeatureGroup):
                PREFIX_PATTERN = f".*__{_CAPTURE}_chsb$"
                PROPERTY_MAPPING = _HANDLE_MAPPING

        assert "ChsBoundPlainGroupB3" in str(exc.value)
        assert HANDLE_KEY in str(exc.value)

    def test_the_same_group_without_the_capture_defines_fine(self) -> None:
        class ChsUnboundMixinGroupB4(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__\w+_chsb4$"
            PROPERTY_MAPPING = _HANDLE_MAPPING

        assert ChsUnboundMixinGroupB4.PROPERTY_MAPPING is _HANDLE_MAPPING

    def test_a_positional_pattern_never_binds_the_handle(self) -> None:
        class ChsPositionalGroupB5(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = r".*__(\w+)_chsb5$"
            PROPERTY_MAPPING = _HANDLE_MAPPING

        assert HANDLE_KEY not in ChsPositionalGroupB5.name_bindings("col__abc_chsb5")

    def test_an_unrelated_named_capture_beside_a_handle_key_defines_fine(self) -> None:
        class ChsOtherCaptureGroupB6(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = r".*__(?P<chs_other_key>\w+)_chsb6$"
            PROPERTY_MAPPING = {
                **_HANDLE_MAPPING,
                "chs_other_key": PropertySpec("Bound from the name.", context=True),
            }

        assert ChsOtherCaptureGroupB6.name_bindings("col__abc_chsb6") == {"chs_other_key": "abc"}
