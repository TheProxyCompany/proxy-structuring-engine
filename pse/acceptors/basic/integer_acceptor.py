from __future__ import annotations

from pse.acceptors.basic.character_acceptor import CharacterAcceptor, CharacterWalker


class IntegerAcceptor(CharacterAcceptor):
    """
    Accepts an integer as per JSON specification.
    """

    def __init__(self, drop_leading_zeros: bool = True) -> None:
        super().__init__("0123456789")
        self.drop_leading_zeros = drop_leading_zeros

    def get_new_walker(self, state: int | str) -> IntegerWalker:
        return IntegerWalker(self)


class IntegerWalker(CharacterWalker):
    """
    Walker for IntegerAcceptor.
    """

    def __init__(self, acceptor: IntegerAcceptor, value: str | None = None) -> None:
        super().__init__(acceptor, value)
        self.acceptor: IntegerAcceptor = acceptor

    @property
    def current_value(self) -> str | None:
        if self.acceptor.drop_leading_zeros:
            return super().parse_value(self._raw_value)
        return self._raw_value
