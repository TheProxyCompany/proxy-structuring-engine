from __future__ import annotations

from typing import Optional, Type
from pse.acceptors.basic.character_acceptor import CharacterAcceptor, CharacterWalker
from pse.core.walker import Walker


class IntegerAcceptor(CharacterAcceptor):
    """
    Accepts an integer as per JSON specification.
    """

    def __init__(self, drop_leading_zeros: bool = True) -> None:
        super().__init__("0123456789")
        self.drop_leading_zeros = drop_leading_zeros

    @property
    def walker_class(self) -> Type[Walker]:
        return IntegerWalker

class IntegerWalker(CharacterWalker):
    """
    Walker for IntegerAcceptor.
    """

    def __init__(self, acceptor: IntegerAcceptor, value: Optional[str] = None) -> None:
        super().__init__(acceptor, value)
        self.acceptor: IntegerAcceptor = acceptor

    @property
    def current_value(self) -> str | None:
        if self.acceptor.drop_leading_zeros:
            return super()._parse_value(self._raw_value)
        return self._raw_value
