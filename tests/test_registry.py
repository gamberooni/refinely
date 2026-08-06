import optuna
import pytest

import apps  # noqa: F401  (registers demo apps)
from crucible.core.exceptions import EvalError
from crucible.dspy import DspyProgramSpec
from crucible.registry import (
    AppRegistration,
    discover_apps,
    get_registration,
    register_app,
    registered_apps,
)

DEMO_APPS = ["extraction", "qa", "rag"]

EXTRACTION_TRIAL = optuna.trial.FixedTrial({"temperature": 0.5, "system_prompt_variant": "verbose"})
QA_TRIAL = optuna.trial.FixedTrial(
    {"temperature": 0.0, "top_k": 4, "system_prompt_variant": "strict"}
)
RAG_TRIAL = optuna.trial.FixedTrial(
    {
        "temperature": 0.5,
        "system_prompt_variant": "verbose",
        "retrieval_strategy": "hybrid",
        "top_k": 4,
        "query_expansion": True,
        "rerank": False,
    }
)


def test_registered_apps_contains_demo_apps_sorted() -> None:
    names = registered_apps()
    assert names == sorted(names)
    assert set(DEMO_APPS) <= set(names)


def test_get_registration_roundtrip() -> None:
    registration = get_registration("extraction")

    assert registration.name == "extraction"
    assert registration.dataset_path.name == "extraction_v1.json"
    assert set(registration.default_config) == {
        "temperature",
        "system_prompt_variant",
    }


def test_get_registration_unknown_app_raises() -> None:
    with pytest.raises(EvalError, match="No app registered: 'nope'"):
        get_registration("nope")


def test_register_app_roundtrip() -> None:
    register_app(
        AppRegistration(
            name="zoo_app",
            build_adapter=lambda client, settings: None,
            metrics_factory=lambda client, settings: [],
            search_space=lambda trial: {},
            default_config={},
            weights={},
        )
    )

    assert "zoo_app" in registered_apps()
    assert get_registration("zoo_app").name == "zoo_app"


def test_register_app_duplicate_raises() -> None:
    with pytest.raises(EvalError, match="App already registered: 'qa'"):
        register_app(
            AppRegistration(
                name="qa",
                build_adapter=lambda client, settings: None,
                metrics_factory=lambda client, settings: [],
                search_space=lambda trial: {},
                default_config={},
                weights={},
            )
        )


def test_registration_dispatch_components_are_callable() -> None:
    for name in DEMO_APPS:
        registration = get_registration(name)
        assert callable(registration.metrics_factory)
        assert callable(registration.build_adapter)
        assert callable(registration.search_space)


def test_search_space_keys_match_default_config() -> None:
    trials = {
        "extraction": EXTRACTION_TRIAL,
        "qa": QA_TRIAL,
        "rag": RAG_TRIAL,
    }
    for name in DEMO_APPS:
        registration = get_registration(name)
        config = registration.search_space(trials[name])
        assert set(config) == set(registration.default_config)


def test_search_space_ranges_and_choices() -> None:
    extraction = get_registration("extraction").search_space(EXTRACTION_TRIAL)
    assert 0.0 <= extraction["temperature"] <= 1.0
    assert extraction["system_prompt_variant"] in {"strict", "verbose"}

    qa = get_registration("qa").search_space(QA_TRIAL)
    assert 0.0 <= qa["temperature"] <= 1.0
    assert 1 <= qa["top_k"] <= 5
    assert qa["system_prompt_variant"] in {"strict", "verbose"}

    rag = get_registration("rag").search_space(RAG_TRIAL)
    assert 0.0 <= rag["temperature"] <= 1.0
    assert 1 <= rag["top_k"] <= 6
    assert rag["retrieval_strategy"] in {"keyword", "hybrid"}
    assert rag["query_expansion"] in {True, False}
    assert rag["rerank"] in {True, False}


def test_weights_sum_to_one_per_app() -> None:
    for name in DEMO_APPS:
        weights = get_registration(name).weights
        assert sum(weights.values()) == pytest.approx(1.0)


def test_register_app_dspy_factory_roundtrip() -> None:
    def factory(settings):
        return DspyProgramSpec(
            build=lambda: object(),
            prepare_example=lambda case: object(),
            prediction_to_output=lambda pred: {},
        )

    register_app(
        AppRegistration(
            name="dspy_app",
            build_adapter=lambda client, settings: None,
            metrics_factory=lambda client, settings: [],
            search_space=lambda trial: {},
            default_config={},
            weights={},
            dspy_factory=factory,
        )
    )

    registration = get_registration("dspy_app")
    assert registration.dspy_factory is factory
    spec = registration.dspy_factory(None)
    assert isinstance(spec, DspyProgramSpec)


def test_register_app_without_dspy_factory_ok() -> None:
    register_app(
        AppRegistration(
            name="plain_app",
            build_adapter=lambda client, settings: None,
            metrics_factory=lambda client, settings: [],
            search_space=lambda trial: {},
            default_config={},
            weights={},
        )
    )

    assert get_registration("plain_app").dspy_factory is None


def _plugin_entry_point(name: str):
    import types

    return types.SimpleNamespace(
        name=name,
        value=f"tests.plugin_{name}",
        load=lambda: register_app(
            AppRegistration(
                name=name,
                build_adapter=lambda client, settings: None,
                metrics_factory=lambda client, settings: [],
                search_space=lambda trial: {},
                default_config={},
                weights={},
            )
        ),
    )


def test_discover_apps_imports_entry_point_modules(monkeypatch) -> None:
    calls: list[str] = []

    def fake_entry_points(*, group: str | None = None):
        calls.append(group)
        return []

    monkeypatch.setattr("importlib.metadata.entry_points", fake_entry_points)

    names = discover_apps()

    assert calls == ["crucible.apps"]
    assert names == sorted(names)
    assert set(DEMO_APPS) <= set(names)


def test_discover_apps_registers_plugins(monkeypatch) -> None:
    ep = _plugin_entry_point("plugin_app")

    def fake_entry_points(*, group: str | None = None):
        assert group == "crucible.apps"
        return [ep]

    monkeypatch.setattr("importlib.metadata.entry_points", fake_entry_points)

    names = discover_apps()

    assert "plugin_app" in names
    assert get_registration("plugin_app").name == "plugin_app"
