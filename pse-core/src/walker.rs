use pyo3::prelude::*;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use trie_rs::Trie;

use crate::acceptor::State;
use crate::acceptor::Acceptor;

#[pyclass(subclass)]
pub struct Walker {
    acceptor: Acceptor,
    current_state: State,
    target_state: Option<State>,
    transition_walker: Option<Box<Walker>>,
    consumed_character_count: usize,
    remaining_input: Option<String>,
    explored_edges: HashSet<(State, Option<State>, Option<String>)>,
    accepted_history: Vec<Walker>,
    _raw_value: Option<String>,
    _accepts_more_input: bool,
}

impl Clone for Walker {
    fn clone(&self) -> Self {
        Self {
            transition_walker: self.transition_walker.clone(),
            acceptor: self.acceptor.clone(),
            current_state: self.current_state.clone(),
            target_state: self.target_state.clone(),
            consumed_character_count: self.consumed_character_count,
            remaining_input: self.remaining_input.clone(),
            explored_edges: self.explored_edges.clone(),
            accepted_history: self.accepted_history.clone(),
            _raw_value: self._raw_value.clone(),
            _accepts_more_input: self._accepts_more_input,
        }
    }
}

impl Walker {
    fn find_valid_prefixes(&self, trie: &Trie<u8>) -> PyResult<HashSet<String>> {
        let mut valid_prefixes = HashSet::new();
        let mut seen = HashSet::new();

        for continuation in self.get_valid_continuations(0)? {
            if seen.contains(&continuation) {
                continue;
            }

            seen.insert(continuation.clone());
            let tokens: Vec<String> = trie.common_prefix_search(continuation).collect();
            valid_prefixes.extend(tokens);
        }

        Ok(valid_prefixes)
    }

}

#[pymethods]
impl Walker {

    #[new]
    #[pyo3(signature = (acceptor, current_state=None))]
    pub fn new(acceptor: Acceptor, current_state: Option<State>) -> PyResult<Self> {
        Ok(Walker {
            acceptor,
            current_state: current_state.unwrap_or(State::Int(0)),
            target_state: None,
            transition_walker: None,
            consumed_character_count: 0,
            remaining_input: None,
            explored_edges: HashSet::new(),
            accepted_history: Vec::new(),
            _raw_value: None,
            _accepts_more_input: false,
        })
    }

    /// Abstract method: consume_token
    fn consume_token(&self, _token: &str) -> PyResult<Vec<Walker>> {
        Ok(vec![]) // Implemented by subclasses
    }

    /// Abstract method: can_accept_more_input
    fn can_accept_more_input(&self) -> PyResult<bool> {
        Ok(self._accepts_more_input)
    }

    /// Abstract method: is_within_value
    fn is_within_value(&self) -> PyResult<bool> {
        Ok(false) // Implemented by subclasses
    }

    #[getter]
    pub fn raw_value(&self) -> PyResult<Option<String>> {
        if self._raw_value.is_some() {
            return Ok(self._raw_value.clone());
        }

        if self.accepted_history.is_empty() && self.transition_walker.is_none() {
            return Ok(None);
        }

        let mut values = Vec::new();

        // Get values from history
        for walker in &self.accepted_history {
            if let Some(val) = walker.raw_value()? {
                values.push(val);
            }
        }

        // Get value from transition walker
        if let Some(walker) = &self.transition_walker {
            if let Some(val) = walker.raw_value()? {
                values.push(val);
            }
        }

        Ok(Some(values.join("")))
    }

    pub fn clone(&self) -> PyResult<Self> {
        let mut clone = <Self as Clone>::clone(self);
        clone.accepted_history = self.accepted_history.clone();
        clone.explored_edges = self.explored_edges.clone();
        Ok(clone)
    }

    pub fn should_start_transition(&mut self, token: &str) -> PyResult<bool> {
        if let Some(walker) = &mut self.transition_walker {
            return walker.should_start_transition(token);
        }

        let current_edge = (
            self.current_state.clone(),
            self.target_state.clone(),
            self.raw_value()?
        );

        if self.explored_edges.contains(&current_edge) {
            self._accepts_more_input = false;
            return Ok(false);
        }

        Ok(true)
    }

    pub fn should_complete_transition(&self) -> PyResult<bool> {
        if let Some(walker) = &self.transition_walker {
            return walker.should_complete_transition();
        }

        Ok(true)
    }

    #[pyo3(signature = (transition_walker, token=None, start_state=None, target_state=None))]
    pub fn start_transition(
        &self,
        mut transition_walker: Walker,
        token: Option<String>,
        start_state: Option<State>,
        target_state: Option<State>
    ) -> PyResult<Option<Walker>> {
        if let Some(t) = &token {
            if !transition_walker.should_start_transition(t)? {
                return Ok(None);
            }
        }

        if self.target_state == target_state
            && self.transition_walker.is_some()
            && self.transition_walker.as_ref().unwrap().can_accept_more_input()? {
            return Ok(None);
        }

        let mut clone = self.clone()?;
        clone.current_state = start_state.unwrap_or(clone.current_state);
        clone.target_state = target_state;

        if let Some(walker) = &clone.transition_walker {
            if walker.has_reached_accept_state()? {
                clone.accepted_history.push(*walker.clone());
            }
        }

        clone.transition_walker = Some(Box::new(transition_walker));

        Ok(Some(clone))
    }

    pub fn has_reached_accept_state(&self) -> PyResult<bool> {
        Ok(false)
    }

    pub fn get_valid_continuations(&self, depth: i32) -> PyResult<Vec<String>> {
        if depth > 10 {
            return Ok(vec![]);
        }

        if let Some(walker) = &self.transition_walker {
            return walker.get_valid_continuations(depth + 1);
        }

        Ok(vec![])
    }

    pub fn accepts_any_token(&self) -> PyResult<bool> {
        if let Some(walker) = &self.transition_walker {
            return walker.accepts_any_token();
        }
        Ok(false)
    }

    #[pyo3(signature = (token=None))]
    pub fn branch(&self, token: Option<String>) -> PyResult<Vec<Walker>> {
        if let Some(walker) = &self.transition_walker {
            let mut transition_walkers = Vec::new();
            if walker.can_accept_more_input()? {
                transition_walkers.extend(walker.branch(token.clone())?);
            }

            let mut results = Vec::new();
            for new_transition_walker in &transition_walkers {
                let mut clone = self.clone()?;
                clone.transition_walker = Some(Box::new(new_transition_walker.clone()?));
                results.push(clone);
            }

            if !transition_walkers.is_empty() || !walker.has_reached_accept_state()? {
                return Ok(results);
            }
        }

        Ok(self.acceptor.branch_walker(self, token)?)
    }

    pub fn complete_transition(&self, transition: Walker) -> PyResult<(Option<Walker>, bool)> {
        let mut clone = self.clone()?;
        clone.transition_walker = Some(Box::new(transition.clone()?));

        clone.remaining_input = transition.remaining_input.clone();
        if let Some(walker) = &mut clone.transition_walker {
            walker.remaining_input = None;
        }

        clone.consumed_character_count += transition.consumed_character_count;
        clone.explored_edges.insert((
            clone.current_state.clone(),
            clone.target_state.clone(),
            clone.raw_value()?,
        ));

        if !clone.should_complete_transition()? {
            return Ok((Some(clone), false));
        }

        if clone.target_state.is_some() && transition.has_reached_accept_state()? {
            clone.current_state = clone.target_state.clone().unwrap();

            if !transition.can_accept_more_input()? {
                clone.accepted_history.push(transition);
                clone.transition_walker = None;
                clone.target_state = None;
            }

            if clone.acceptor.end_states.contains(&clone.current_state) {
                return Ok((Some(clone), true));
            }
        }

        Ok((Some(clone), false))
    }
}

impl Hash for Walker {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.current_state.hash(state);
        if let Some(target) = &self.target_state {
            target.hash(state);
        }
        if let Some(raw) = &self._raw_value {
            raw.hash(state);
        }
        if let Some(walker) = &self.transition_walker {
            walker.hash(state);
        }
    }
}

impl PartialEq for Walker {
    fn eq(&self, other: &Self) -> bool {
        self.current_state == other.current_state
            && self.target_state == other.target_state
            && self.raw_value().ok() == other.raw_value().ok()
            && self.transition_walker == other.transition_walker
    }
}

impl Eq for Walker {}
