from typing import Optional

class JSONParsingError(Exception):
    """
    Custom exception for JSON parsing errors.
    """

    def __init__(
        self,
        message: str,
        position: Optional[int] = None,
        value: Optional[str] = None,
    ):
        super().__init__(message)
        self.position = position
        self.value = value

class SchemaNotImplementedError(JSONParsingError):
    """
    Raised when a JSON schema uses a feature that hasn't been implemented here yet
    """

class InvalidSchemaError(JSONParsingError):
    """
    Raised when the passed JSON schema is invalid, e.g. a required property is not defined
    """

class UnknownSchemaTypeError(JSONParsingError):
    """
    Raised when a JSON schema doesn't contain a known type.
    """

class TokenRejected(JSONParsingError):
    """
    Raised when the token cannot advance any of the current acceptors.
    """

class DefinitionNotFoundError(JSONParsingError):
    """
    Raised when a JSON schema reference cannot be resolved.
    """
