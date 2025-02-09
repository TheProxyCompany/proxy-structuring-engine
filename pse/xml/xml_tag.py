from pse.base.chain import ChainStateMachine
from pse.base.phrase import PhraseStateMachine


class XMLTagStateMachine(ChainStateMachine):
    """
    A state machine that recognizes XML tags.
    """

    def __init__(self, tag_name: str, closing_tag: bool = False) -> None:
        self.tag_name = tag_name
        super().__init__(
            [
                PhraseStateMachine("<" if not closing_tag else "</"),
                PhraseStateMachine(tag_name),
                PhraseStateMachine(">"),
            ]
        )

    def __str__(self) -> str:
        return self.tag_name
