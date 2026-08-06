"""Wire DSPy's language model to crucible's `Settings` + OpenAI-compatible gateway."""

from typing import Any

from crucible.core.settings import Settings
from crucible.dspy._imports import _dspy


def configure_lm(settings: Settings, temperature: float = 0.0, **kwargs: Any) -> Any:
    """Create a `dspy.LM` from `Settings` and call `dspy.configure(lm=lm)`.

    The provider prefix is always ``openai/`` because the model is served by an
    OpenAI-compatible gateway; `api_base` is forwarded only when `base_url` is
    set so direct OpenAI runs keep the default endpoint.
    """
    dspy = _dspy()
    model = f"openai/{settings.model_name}"
    lm_kwargs: dict[str, Any] = {
        "temperature": temperature,
        "api_key": settings.openai_api_key,
        **kwargs,
    }
    if settings.base_url:
        lm_kwargs["api_base"] = settings.base_url
    lm = dspy.LM(model, **lm_kwargs)
    dspy.configure(lm=lm)
    return lm
