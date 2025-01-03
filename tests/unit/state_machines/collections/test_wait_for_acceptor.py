from pse_core.trie import TrieSet

from pse.state_machines.basic.text_acceptor import TextAcceptor, TextWalker
from pse.state_machines.collections.wait_for_acceptor import (
    WaitForAcceptor,
    WaitForWalker,
)


def test_default_wait_for_acceptor() -> None:
    text_acceptor = TextAcceptor("Hello World")
    state_machine = WaitForAcceptor(text_acceptor)

    walkers = list(state_machine.get_walkers())
    assert len(walkers) == 1
    walker = walkers[0]
    assert isinstance(walker, WaitForWalker)
    assert walker.accepts_any_token()
    assert walker.transition_walker
    assert walker.transition_walker.state_machine == text_acceptor
    assert isinstance(walker.transition_walker, TextWalker)
    assert not walker.is_within_value()
    assert not walker.can_accept_more_input()


def test_basic_wait_for_acceptor() -> None:
    """Test that the WaitForAcceptor can accept any token."""
    text_acceptor = TextAcceptor("Hello World")
    state_machine = WaitForAcceptor(text_acceptor)
    walkers = list(state_machine.get_walkers())
    walkers = [
        walker for _, walker in state_machine.advance_all(walkers, "Hello World")
    ]
    assert len(walkers) == 1
    assert walkers[0].has_reached_accept_state()


def test_interrupted_wait_for_acceptor() -> None:
    text_acceptor = TextAcceptor("Hello World")
    state_machine = WaitForAcceptor(text_acceptor)

    walkers = list(state_machine.get_walkers())
    walkers = [walker for _, walker in state_machine.advance_all(walkers, "Hello ")]
    assert len(walkers) == 1
    assert walkers[0].is_within_value()
    walkers = [
        walker
        for _, walker in state_machine.advance_all(
            walkers, "I'm gonna mess up the pattern! But i'll still be accepted!"
        )
    ]
    assert len(walkers) == 1
    assert not walkers[0].is_within_value()

    walkers = [
        walker for _, walker in state_machine.advance_all(walkers, "Hello World")
    ]
    assert len(walkers) == 1
    assert walkers[0].has_reached_accept_state()


def test_wait_for_acceptor_with_break() -> None:
    """Test that the WaitForAcceptor can accept any token."""
    text_acceptor = TextAcceptor("Hello World")
    state_machine = WaitForAcceptor(text_acceptor, allow_break=True)
    walkers = list(state_machine.get_walkers())
    walkers = [walker for _, walker in state_machine.advance_all(walkers, "Hello ")]
    assert len(walkers) == 1

    walkers = [
        walker
        for _, walker in state_machine.advance_all(
            walkers, "I'm gonna mess up the pattern! But i'll still be accepted!"
        )
    ]
    assert len(walkers) == 1

    walkers = [walker for _, walker in state_machine.advance_all(walkers, "World")]
    assert len(walkers) == 1
    assert walkers[0].has_reached_accept_state()

def test_wait_for_acceptor_with_partial_match():
    """Test that the WaitForAcceptor can accept any token."""
    text_acceptor = TextAcceptor('"hello"')
    state_machine = WaitForAcceptor(text_acceptor)
    walkers = list(state_machine.get_walkers())
    trie_set = TrieSet()
    keys = ['"hello', '"', "hello", '"c']
    trie_set = trie_set.insert_all(keys)
    for advanced_token, walker in state_machine.advance_all(walkers, '"*', trie_set):
        assert walker.get_current_value() == '"'
        assert advanced_token == '"'
    assert len(walkers) == 1
    assert not walkers[0].has_reached_accept_state()
