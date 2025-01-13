from __future__ import annotations

from pse_core.state_machine import StateMachine

from pse.state_machines.base.phrase import PhraseStateMachine


class BooleanStateMachine(StateMachine):
    """
    Accepts a JSON boolean value: true, false.
    """

    def __init__(self) -> None:
        """
        Initialize the BooleanAcceptor with its state transitions defined as a state graph.
        """
        super().__init__(
            {
                0: [
                    (PhraseStateMachine("true"), "$"),
                    (PhraseStateMachine("false"), "$"),
                ]
            }
        )
