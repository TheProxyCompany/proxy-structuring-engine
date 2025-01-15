from pse_core.trie import TrieSet

from pse.state_machines.base.phrase import PhraseStateMachine, PhraseWalker
from pse.state_machines.composite.wait_for import (
    WaitForStateMachine,
    WaitForWalker,
)


def test_default_wait_for_acceptor() -> None:
    text_acceptor = PhraseStateMachine("Hello World")
    state_machine = WaitForStateMachine(text_acceptor)

    walkers = list(state_machine.get_walkers())
    assert len(walkers) == 1
    walker = walkers[0]
    assert isinstance(walker, WaitForWalker)
    assert walker.accepts_any_token()
    assert walker.transition_walker
    assert walker.transition_walker.state_machine == text_acceptor
    assert isinstance(walker.transition_walker, PhraseWalker)
    assert not walker.is_within_value()
    walkers = [walker for _, walker in state_machine.advance_all(walkers, "Hello ")]
    assert len(walkers) == 1
    assert walkers[0].is_within_value()
    walkers = [walker for _, walker in state_machine.advance_all(walkers, "World")]
    assert len(walkers) == 1
    assert walkers[0].has_reached_accept_state()


def test_basic_wait_for_acceptor() -> None:
    """Test that the WaitForAcceptor can accept any token."""
    text_acceptor = PhraseStateMachine("Hello World")
    state_machine = WaitForStateMachine(text_acceptor)
    walkers = list(state_machine.get_walkers())
    walkers = [
        walker for _, walker in state_machine.advance_all(walkers, "Hello World")
    ]
    assert len(walkers) == 1
    assert walkers[0].has_reached_accept_state()


def test_interrupted_wait_for_acceptor() -> None:
    text_acceptor = PhraseStateMachine("Hello World")
    state_machine = WaitForStateMachine(text_acceptor)

    walkers = state_machine.get_walkers()
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
    text_acceptor = PhraseStateMachine("Hello World")
    state_machine = WaitForStateMachine(text_acceptor, allow_break=True)
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
    text_acceptor = PhraseStateMachine('"hello"')
    state_machine = WaitForStateMachine(text_acceptor)
    walkers = list(state_machine.get_walkers())
    trie_set = TrieSet()
    keys = ['"hello', '"', "hello", '"c']
    trie_set = trie_set.insert_all(keys)
    for advanced_token, walker in state_machine.advance_all(walkers, '"*', trie_set):
        assert walker.get_current_value() == '"'
        assert advanced_token == '"'
    assert len(walkers) == 1
    assert not walkers[0].has_reached_accept_state()
