// // pse-core/src/state_machine.rs

// //! A hierarchical state machine implementation for token-based parsing and validation.
// //!
// //! This module provides a flexible state machine framework that:
// //! - Supports parallel recursive descent parsing
// //! - Enables efficient graph-based token acceptance
// //! - Handles branching and backtracking through parallel walker exploration
// //! - Allows composition of sub-state machines for complex grammars
// //! - Provides case-sensitive and case-insensitive matching options

// use pyo3::prelude::*;
// use pyo3::types::{PyAny, PyList, PyString};
// use std::collections::{HashMap, HashSet, VecDeque};
// use std::sync::Arc;
// use log::{debug, info, warn};

// use crate::acceptor::{Acceptor, Edge, State};
// use crate::walker::{Walker, WalkerBehavior};

// // Define the StateMachine struct, extending the Acceptor
// #[pyclass(extends=Acceptor)]
// pub struct StateMachine {
//     /// The walker class associated with this state machine
//     #[pyo3(get)]
//     walker_class: Py<PyType>,
// }

// #[pymethods]
// impl StateMachine {
//     /// Creates a new StateMachine instance
//     #[new]
//     #[args(
//         state_graph = "None",
//         start_state = "None",
//         end_states = "None",
//         is_optional = "false",
//         is_case_sensitive = "true"
//     )]
//     pub fn new(
//         py: Python,
//         state_graph: Option<HashMap<State, Vec<Edge>>>,
//         start_state: Option<State>,
//         end_states: Option<Vec<State>>,
//         is_optional: bool,
//         is_case_sensitive: bool,
//     ) -> PyResult<(Self, Acceptor)> {
//         let acceptor = Acceptor::new(
//             py,
//             state_graph,
//             start_state,
//             end_states,
//             is_optional,
//             is_case_sensitive,
//         )?;
//         let walker_class = py.get_type::<StateMachineWalker>().into();
//         Ok((
//             StateMachine { walker_class },
//             acceptor,
//         ))
//     }

//     /// Retrieves outgoing transitions for a given state
//     #[pyo3(name = "get_edges")]
//     pub fn get_edges(&self, py: Python, state: State) -> PyResult<Vec<Edge>> {
//         let acceptor = py
//             .extract::<PyRef<Acceptor>>(self.as_ref(py))?;
//         Ok(acceptor
//             .state_graph
//             .get(&state)
//             .cloned()
//             .unwrap_or_default())
//     }

//     /// Initializes walkers at the specified start state
//     #[pyo3(name = "get_walkers")]
//     pub fn get_walkers(&self, py: Python, state: Option<State>) -> PyResult<Py<PyList>> {
//         let acceptor = py
//             .extract::<PyRef<Acceptor>>(self.as_ref(py))?;
//         let initial_state = state.unwrap_or_else(|| acceptor.start_state.clone());

//         let walker_class = self.walker_class.as_ref(py);
//         let initial_walker = walker_class.call1((self.clone_ref(py), initial_state))?;
//         let mut walkers = Vec::new();

//         if !acceptor.state_graph.is_empty() {
//             walkers.extend(self.branch_walker(py, initial_walker, None)?);
//         } else {
//             walkers.push(initial_walker);
//         }

//         Ok(PyList::new(py, walkers).into())
//     }

//     /// Retrieves transition walkers from the current state
//     #[pyo3(name = "get_transitions")]
//     pub fn get_transitions(
//         &self,
//         py: Python,
//         walker: &PyAny,
//         state: Option<State>,
//     ) -> PyResult<Vec<(PyObject, State, State)>> {
//         let acceptor = py
//             .extract::<PyRef<Acceptor>>(self.as_ref(py))?;
//         let current_state = state.unwrap_or_else(|| walker.getattr("current_state").unwrap().extract().unwrap());
//         let mut transitions = Vec::new();

//         if let Some(edges) = acceptor.state_graph.get(&current_state) {
//             for (acceptor_obj, target_state) in edges {
//                 let acceptor_instance = acceptor_obj.clone_ref(py);
//                 let acceptor_py = acceptor_instance.as_ref(py);
//                 let acceptor_walkers = acceptor_py.call_method0("get_walkers")?.extract::<&PyList>()?;
//                 for transition in acceptor_walkers.iter() {
//                     transitions.push((
//                         transition.into(),
//                         current_state.clone(),
//                         target_state.clone(),
//                     ));
//                 }

//                 let is_optional: bool = acceptor_py.getattr("is_optional")?.extract()?;
//                 let end_states = &acceptor.end_states;
//                 let can_accept_more_input: bool = walker.call_method0("can_accept_more_input")?.extract()?;
//                 if is_optional && !end_states.contains(target_state) && can_accept_more_input {
//                     debug!(
//                         "� {:?} supports pass-through to state {:?}",
//                         acceptor_py, target_state
//                     );
//                     let sub_transitions = self.get_transitions(py, walker, Some(target_state.clone()))?;
//                     transitions.extend(sub_transitions);
//                 }
//             }
//         }

//         Ok(transitions)
//     }

//     /// Branches the walker into multiple paths for parallel exploration
//     #[pyo3(name = "branch_walker")]
//     pub fn branch_walker(
//         &self,
//         py: Python,
//         walker: Py<PyAny>,
//         token: Option<String>,
//     ) -> PyResult<Vec<PyObject>> {
//         debug!("🔵 Branching {:?}", walker.as_ref(py));
//         let input_token = token.or_else(|| {
//             walker
//                 .as_ref(py)
//                 .getattr("remaining_input")
//                 .ok()
//                 .and_then(|obj| obj.extract::<Option<String>>().ok())
//                 .flatten()
//         });

//         let transitions = self.get_transitions(py, walker.as_ref(py), None)?;
//         let mut branched_walkers = Vec::new();

//         for (transition, start_state, target_state) in transitions {
//             let start_transition = walker
//                 .as_ref(py)
//                 .call_method(
//                     "start_transition",
//                     (
//                         transition.clone(),
//                         input_token.clone(),
//                         start_state.clone(),
//                         target_state.clone(),
//                     ),
//                     None,
//                 );
//             if let Ok(branched_walker) = start_transition {
//                 branched_walkers.push(branched_walker);
//                 continue;
//             }

//             let acceptor = transition.as_ref(py).getattr("acceptor")?;
//             let is_optional: bool = acceptor.getattr("is_optional")?.extract()?;
//             let acceptor_end_states = py
//                 .extract::<PyRef<Acceptor>>(self.as_ref(py))?
//                 .end_states;
//             if is_optional
//                 && acceptor_end_states.contains(&target_state)
//                 && input_token.is_some()
//             {
//                 debug!("🟠 {:?} is optional; yielding accepted state", transition);
//                 let remaining_input = walker
//                     .as_ref(py)
//                     .getattr("remaining_input")?
//                     .extract::<Option<String>>()?;
//                 if remaining_input.is_none() {
//                     walker.as_ref(py).setattr("remaining_input", input_token.clone())?;
//                 }
//                 let accepted_state = py.get_type::<AcceptedState>().call1((walker.clone_ref(py),))?;
//                 branched_walkers.push(accepted_state);
//             }
//         }

//         Ok(branched_walkers)
//     }

//     /// Processes a token through the state machine, advancing walker states and managing transitions
//     #[pyo3(name = "advance")]
//     pub fn advance(
//         &self,
//         py: Python,
//         walker: Py<PyAny>,
//         input_token: String,
//     ) -> PyResult<Vec<PyObject>> {
//         let mut queue: VecDeque<(Py<PyAny>, String)> = VecDeque::new();
//         queue.push_back((walker, input_token));

//         let mut results = Vec::new();

//         while let Some((current_walker, current_token)) = queue.pop_front() {
//             let should_start_transition: bool = current_walker
//                 .as_ref(py)
//                 .call_method1("should_start_transition", (current_token.clone(),))?
//                 .extract()?;

//             let transition_walker_exists = current_walker
//                 .as_ref(py)
//                 .getattr("transition_walker")?
//                 .is_none();

//             if !should_start_transition || transition_walker_exists {
//                 results.extend(self.handle_blocked_transition(py, current_walker.clone(), current_token.clone())?);
//                 continue;
//             }

//             let consume_token_result = current_walker
//                 .as_ref(py)
//                 .call_method1("consume_token", (current_token.clone(),))?;

//             let consumed_walkers = consume_token_result.extract::<Vec<PyObject>>()?;
//             for transitioned_walker in consumed_walkers {
//                 let remaining_input = transitioned_walker
//                     .as_ref(py)
//                     .getattr("remaining_input")?
//                     .extract::<Option<String>>()?;
//                 if remaining_input.is_some() {
//                     queue.push_back((transitioned_walker.clone_ref(py), remaining_input.unwrap()));
//                 } else {
//                     results.push(transitioned_walker);
//                 }
//             }
//         }

//         Ok(results)
//     }

//     fn handle_blocked_transition(
//         &self,
//         py: Python,
//         blocked_walker: Py<PyAny>,
//         token: String,
//     ) -> PyResult<Vec<PyObject>> {
//         let branch_method = blocked_walker.call_method1("branch", (token.clone(),))?;
//         let branched_walkers = branch_method.extract::<Vec<PyObject>>()?;
//         let mut queue = Vec::new();

//         for branched_walker in &branched_walkers {
//             let should_start_transition: bool = branched_walker
//                 .as_ref(py)
//                 .call_method1("should_start_transition", (token.clone(),))?
//                 .extract()?;
//             if should_start_transition {
//                 queue.push((branched_walker.clone_ref(py), token.clone()));
//             } else if branched_walker
//                 .as_ref(py)
//                 .call_method0("has_reached_accept_state")?
//                 .extract::<bool>()?
//             {
//                 debug!("🟠 Walker has reached accept state: {:?}", branched_walker);
//                 return Ok(vec![branched_walker.clone_ref(py)]);
//             }
//         }

//         if queue.is_empty() && blocked_walker.getattr("remaining_input")?.extract::<Option<String>>()?.is_some() {
//             debug!("🟠 Walker has remaining input: {:?}", blocked_walker);
//             return Ok(vec![blocked_walker.clone()]);
//         } else if queue.is_empty() {
//             debug!("🔴 {:?} cannot parse {:?}", blocked_walker, token);
//         }

//         Ok(vec![])
//     }

//     /// Advances all walkers to find valid token matches
//     #[staticmethod]
//     pub fn advance_all(
//         py: Python,
//         walkers: &PyAny,
//         token: String,
//         vocab: Option<&PyAny>,
//     ) -> PyResult<Vec<(String, PyObject)>> {
//         let mut results = Vec::new();

//         for walker in walkers.iter()? {
//             let walker = walker?;
//             let consumed_walkers = walker.call_method1("consume_token", (token.clone(),))?;
//             for advanced_walker in consumed_walkers.extract::<Vec<PyObject>>()? {
//                 let remaining_input = advanced_walker
//                     .as_ref(py)
//                     .getattr("remaining_input")?
//                     .extract::<Option<String>>()?;
//                 if remaining_input.is_none() {
//                     debug!("🟢 Full match for token: {:?}", token);
//                     results.push((token.clone(), advanced_walker));
//                     continue;
//                 }

//                 if vocab.is_none() {
//                     debug!("🔴 No vocab - unable to check for partial match");
//                     continue;
//                 }

//                 let prefix_length = token.len() - remaining_input.clone().unwrap_or_default().len();
//                 let prefix = &token[..prefix_length];
//                 if !prefix.is_empty() && vocab.unwrap().contains(prefix)? {
//                     debug!("🟢 Valid partial match: {:?}", prefix);
//                     advanced_walker.as_ref(py).setattr("remaining_input", py.None())?;
//                     let can_accept_more_input = advanced_walker
//                         .as_ref(py)
//                         .call_method0("can_accept_more_input")?
//                         .extract::<bool>()?;
//                     let transition_walker_exists = advanced_walker
//                         .as_ref(py)
//                         .getattr("transition_walker")?
//                         .is_none();

//                     if !transition_walker_exists && can_accept_more_input {
//                         let next_walkers = advanced_walker.call_method0("branch")?;
//                         for next_walker in next_walkers.extract::<Vec<PyObject>>()? {
//                             results.push((prefix.to_string(), next_walker));
//                         }
//                     } else {
//                         results.push((prefix.to_string(), advanced_walker));
//                     }
//                 }
//             }
//         }

//         Ok(results)
//     }
// }
