from pse.acceptors.collections.wait_for_acceptor import WaitForAcceptor, WaitForWalker
from pse.acceptors.basic.text_acceptor import TextAcceptor, TextWalker
from pse.core.state_machine import StateMachine


def test_basic_wait_for_acceptor() -> None:
    """Test that the WaitForAcceptor can accept any token."""
    text_acceptor = TextAcceptor("Hello World")
    acceptor = WaitForAcceptor(text_acceptor)

    walkers = list(acceptor.get_walkers())
    assert len(walkers) == 1
    walker = walkers[0]
    assert isinstance(walker, WaitForWalker)
    assert walker.accepts_any_token()
    assert walker.transition_walker
    assert walker.transition_walker.acceptor == text_acceptor
    assert isinstance(walker.transition_walker, TextWalker)
