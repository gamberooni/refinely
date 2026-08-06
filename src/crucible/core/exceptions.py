class CrucibleError(Exception):
    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause


class LLMError(CrucibleError): ...


class EvalError(CrucibleError): ...
