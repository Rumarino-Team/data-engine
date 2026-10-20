from typing import Optional


class OperationError(Exception):
    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        code: Optional[str] = None,
        detail: Optional[str] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.detail = detail
        self.code = code or self._default_code(status_code)

    @staticmethod
    def _default_code(status_code: int) -> str:
        if status_code == 507:
            return "cuda_out_of_memory"
        if status_code in {400, 404, 409}:
            return "validation_error"
        return "backend_error"
