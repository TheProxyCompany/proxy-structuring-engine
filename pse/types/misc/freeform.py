from __future__ import annotations

from pse.types.base.wait_for import WaitFor
from pse.types.enum import EnumStateMachine


class FreeformStateMachine(WaitFor):
    """
    A state machine that can be used to parse freeform text that has an ending delimiter.
    """

    def __init__(
        self,
        end_delimiters: list[str],
        char_min: int | None = None,
    ):
        self.end_delimiters = end_delimiters
        delimiter_state_machine = EnumStateMachine(self.end_delimiters, require_quotes=False)
        super().__init__(
            delimiter_state_machine,
            buffer_length=char_min or 1,
        )
