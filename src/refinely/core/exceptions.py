class RefinelyError(Exception):
    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause


class LLMError(RefinelyError): ...


class EvalError(RefinelyError): ...
