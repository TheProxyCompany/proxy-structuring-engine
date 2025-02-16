from __future__ import annotations

from pse.types.base.chain import ChainStateMachine, ChainStepper
from pse.types.base.phrase import PhraseStateMachine


class XMLTagStateMachine(ChainStateMachine):
    """
    A state machine that recognizes XML tags.
    """

    def __init__(self, tag_name: str, closing_tag: bool = False) -> None:
        self.tag_name = tag_name
        self.xml_tag = ("<" if not closing_tag else "</") + tag_name + ">"
        super().__init__(
            [
                PhraseStateMachine("<" if not closing_tag else "</"),
                PhraseStateMachine(tag_name),
                PhraseStateMachine(">"),
            ]
        )

    def get_new_stepper(self, state: int | str | None = None) -> XMLTagStepper:
        return XMLTagStepper(self, state)

    def __str__(self) -> str:
        return self.tag_name

class XMLTagStepper(ChainStepper):

    def __init__(self, state_machine: XMLTagStateMachine, *args, **kwargs) -> None:
        super().__init__(state_machine, *args, **kwargs)
        self.state_machine: XMLTagStateMachine = state_machine

    def get_valid_continuations(self) -> list[str]:
        return [self.state_machine.xml_tag]
