from typing import Dict, List, Tuple, Union

from pse.acceptors.token_acceptor import TokenAcceptor

StateType = Union[int, str]
"""A state identifier in the StateMachine, which can be an integer or a string."""

# Type alias for edges in the StateMachine graph.
EdgeType = Tuple[TokenAcceptor, StateType]
"""An edge in the StateMachine graph, represented as a tuple of a TokenAcceptor and a
target StateType."""

# Type alias for visited edges in the StateMachine graph.
VisitedEdgeType = Tuple[StateType, StateType, str]
"""An edge in the StateMachine graph, represented as a tuple of a source state, a
target state, and a string representing the value traversed."""

# Type alias for the graph representation of the StateMachine.
StateMachineGraph = Dict[StateType, List[EdgeType]]
"""The graph representation of the StateMachine, mapping each state to a list of its
outgoing edges."""
