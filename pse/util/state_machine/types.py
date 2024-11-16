from typing import Dict, List, Tuple, Union
from pse.acceptors.basic.acceptor import Acceptor

StateType = Union[int, str]
"""Represents a state within the StateMachine.  Can be an integer or a string."""

# Type alias for edges in the StateMachine graph.
EdgeType = Tuple[Acceptor, StateType]
"""Represents a transition between states in the StateMachine.

An Edge consists of a `TokenAcceptor` that governs the conditions for the transition,
and the `StateType` of the destination state.  Walkers traverse edges based on
the acceptance criteria of the associated `TokenAcceptor`.
"""

# Type alias for visited edges in the StateMachine graph.
VisitedEdgeType = Tuple[StateType, StateType, str]
"""Represents a traversed edge during StateMachine execution.

A `VisitedEdgeType` records the source state, target state, and the token
consumed during the transition.  This history aids in tracking walker progress
and preventing cycles.
"""

# Type alias for the graph representation of the StateMachine.
StateMachineGraph = Dict[StateType, List[EdgeType]]
"""Represents the structure of the StateMachine as a directed graph.

Maps each `StateType` to a list of its outgoing `EdgeType`s.  This graph
defines the possible transitions between states and forms the basis for
walker navigation.
"""
