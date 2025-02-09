from pse_core import StateId
from pse_core.state_machine import StateMachine

from pse.base.encapsulated import EncapsulatedStepper
from pse.base.wait_for import WaitFor
from pse.xml.xml_tag import XMLTagStateMachine


class XMLEncapsulatedStateMachine(StateMachine):
    """
    A state machine that wraps a state machine in XML tags.
    """

    def __init__(
        self,
        state_machine: StateMachine,
        tag_name: str,
        min_buffer_length: int = -1,
    ) -> None:
        """

        Args:
            state_machine: The state_machine wrapped by this state machine.
            tag_name: The name of the tag to wrap the state machine in.
        """
        self.inner_state_machine = state_machine
        super().__init__(
            {
                0: [
                    (
                        WaitFor(
                            XMLTagStateMachine(tag_name),
                            min_buffer_length=min_buffer_length,
                        ),
                        1,
                    ),
                ],
                1: [(state_machine, 2)],
                2: [(XMLTagStateMachine(tag_name, closing_tag=True), "$")],
            }
        )

    def get_new_stepper(self, state: StateId | None = None) -> EncapsulatedStepper:
        return EncapsulatedStepper(self, state)
