import logging

from pse_core import StateGraph
from pse_core.state_machine import StateMachine

from pse.base.encapsulated import EncapsulatedStateMachine
from pse.grammar.grammar import GrammarStateMachine
from pse.grammar.python import PythonGrammar
from pse.json import JSONSchemaSource, schema_state_machine

logger = logging.getLogger(__name__)


class StructuringMachine(StateMachine):
    """
    A state machine that can validate both JSON and Python syntax structures.

    This machine acts as a top-level controller that can handle multiple syntax validation
    paths simultaneously. It maintains a "scratchpad" state from which it can transition
    to either JSON or Python validation paths.

    Args:
        json_schemable: Source for JSON schema validation
        delimiters: Optional tuple of (start, end) delimiters for encapsulation
        min_buffer_length: Minimum buffer length before validation starts (-1 for immediate)
        include_python: Whether to include Python grammar validation path
    """

    def __init__(
        self,
        json_schemable: JSONSchemaSource,
        json_delimiters: tuple[str, str] | None = None,
        min_buffer_length: int = -1,
        include_python: bool = False,
    ) -> None:
        end_states = []
        start_state = "scratchpad"
        state_graph: StateGraph = {"scratchpad": []}
        if json_schemable:
            self.json_delimiters = json_delimiters
            end_states += ["json"]
            json_edge = schema_state_machine(
                json_schemable,
                json_delimiters,
                min_buffer_length,
            )
            state_graph["scratchpad"].append((json_edge, "json"))

        if include_python:
            end_states += ["python"]
            python_edge = EncapsulatedStateMachine(
                GrammarStateMachine(PythonGrammar),
                delimiters=PythonGrammar.delimiters,
                min_buffer_length=min_buffer_length,
            )
            state_graph["scratchpad"].append((python_edge, "python"))

        super().__init__(
            state_graph=state_graph,
            start_state=start_state,
            end_states=end_states,
        )
