from __future__ import annotations

from pse_core import StateId

from pse.types.base.character import CharacterStateMachine
from pse.types.base.encapsulated import EncapsulatedStateMachine, EncapsulatedStepper


class FencedFreeformStateMachine(EncapsulatedStateMachine):
    """
    A state machine that can be used to parse freeform text that is enclosed in a pair of delimiters.
    """
    def __init__(self,
                 identifier: str | None = None,
                 delimiter: tuple[str, str] | None = None,
                 buffer_length: int = -1,
                 char_min: int = 1,
                 char_max: int = -1,
                 is_optional: bool = False):

        if delimiter is None:
            delimiter = (f'```{identifier or ""}\n', '\n```')

        freeform_state_machine = CharacterStateMachine(
            whitelist_charset="",
            graylist_charset=set(delimiter[1]),
            blacklist_charset=delimiter[0][0],
            char_min=char_min,
            char_limit=char_max,
        )
        super().__init__(freeform_state_machine, delimiter, buffer_length, is_optional)
        self.identifier = identifier

    def get_new_stepper(self, state: StateId | None = None) -> FencedFreeformStepper:
        return FencedFreeformStepper(self, state)

class FencedFreeformStepper(EncapsulatedStepper):

    def __init__(
        self,
        state_machine: FencedFreeformStateMachine,
        state: StateId | None = None,
    ) -> None:
        super().__init__(state_machine, state)
        self.state_machine: FencedFreeformStateMachine = state_machine

    def get_identifier(self) -> str | None:
        return self.state_machine.identifier

    def get_invalid_continuations(self) -> list[str]:
        if not self.inner_stepper:
            return [self.state_machine.delimiters[1]]
        return super().get_invalid_continuations()
