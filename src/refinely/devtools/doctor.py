"""Health checks for the refinely development environment.

`run_checks` runs deterministic, network-free checks (app discovery, dataset
loading, DB schema, env key). A network probe of the configured gateway runs
only when `--network` is requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.request import urlopen

from refinely.core.settings import Settings
from refinely.eval.datasets import load_dataset
from refinely.registry import discover_apps, registered_apps
from refinely.tracking.db import LineageDB

DEFAULT_OPENAI_ENDPOINT = "https://api.openai.com/v1/models"


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    hint: str = ""


def _check_apps() -> CheckResult:
    apps = discover_apps()
    if not apps:
        return CheckResult(
            "apps", False, "no apps discovered",
            "declare an entry point under [project.entry-points.\"refinely.apps\"] "
            "in pyproject.toml, or run 'refinely new app <name>'",
        )
    return CheckResult("apps", True, f"discovered {len(apps)} app(s): {', '.join(apps)}")


def _check_datasets() -> CheckResult:
    apps = registered_apps()
    failures: list[str] = []
    for app in apps:
        from refinely.registry import get_registration

        reg = get_registration(app)
        try:
            load_dataset(reg.dataset_path)
        except Exception as e:  # noqa: BLE001 - collect all dataset problems
            failures.append(f"{app}: {e}")
    if failures:
        return CheckResult(
            "datasets", False, "; ".join(failures),
            "fix the dataset files under datasets/ (each must parse as a list of cases)",
        )
    return CheckResult("datasets", True, f"loaded {len(apps)} dataset(s) for all registered apps")


def _check_schema(settings: Settings) -> CheckResult:
    try:
        with LineageDB(settings.lineage_db_path) as db:
            db.init_schema()
        return CheckResult("schema", True, f"lineage schema is current ({settings.lineage_db_path})")
    except Exception as e:  # noqa: BLE001
        return CheckResult(
            "schema", False, f"could not initialize lineage DB: {e}",
            "check the lineage.db path is writable and not corrupt",
        )


def _check_env(settings: Settings) -> CheckResult:
    if settings.has_api_key:
        return CheckResult("env", True, "API key is set")
    return CheckResult(
        "env", False, "REFINELY_OPENAI_API_KEY is not set",
        "set REFINELY_OPENAI_API_KEY in the environment or add a .env file",
    )


def _check_network(settings: Settings) -> CheckResult:
    target = settings.base_url or DEFAULT_OPENAI_ENDPOINT
    try:
        with urlopen(target, timeout=5):  # noqa: S310 - user opted into the probe
            pass
        return CheckResult("network", True, f"reachable: {target}")
    except Exception as e:  # noqa: BLE001
        return CheckResult(
            "network", False, f"could not reach {target}: {e}",
            "check your gateway/base_url or network connection",
        )


def run_checks(settings: Settings, network: bool = False) -> list[CheckResult]:
    """Run all checks; the network probe runs only when `network` is True."""
    results = [
        _check_apps(),
        _check_datasets(),
        _check_schema(settings),
        _check_env(settings),
    ]
    if network:
        results.append(_check_network(settings))
    return results
