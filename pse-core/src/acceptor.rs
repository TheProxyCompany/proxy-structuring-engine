use pyo3::prelude::*;
use std::collections::HashMap;
use crate::walker::Walker;

#[derive(Clone, Hash, PartialEq, Eq, FromPyObject, Debug)]
pub enum State {
    Int(usize),
    Str(String),
}
pub type Edge = (Acceptor, State);
pub type StateGraph = HashMap<State, Vec<Edge>>;

#[pyclass(name = "Acceptor", subclass)]
#[derive(Clone, PartialEq)]
pub struct Acceptor {
    state_graph: StateGraph,
    start_state: State,
    pub end_states: Vec<State>,
    is_optional: bool,
    is_case_sensitive: bool,
}

#[pymethods]
impl Acceptor {

    #[new]
    #[pyo3(signature = (
        state_graph=None,
        start_state=State::Int(0),
        end_states=None,
        is_optional=false,
        is_case_sensitive=true
    ))]
    pub fn new(
        state_graph: Option<StateGraph>,
        start_state: State,
        end_states: Option<Vec<State>>,
        is_optional: bool,
        is_case_sensitive: bool,
    ) -> Self {
        let end_states = end_states.unwrap_or_else(|| vec![State::Str("$".to_string())]);
        let state_graph = state_graph.unwrap_or_default();

        Self {
            state_graph,
            start_state,
            end_states,
            is_optional,
            is_case_sensitive,
        }
    }

    #[getter]
    pub fn is_optional(&self) -> bool {
        self.is_optional
    }

    #[getter]
    pub fn is_case_sensitive(&self) -> bool {
        self.is_case_sensitive
    }

    #[pyo3(signature = (_walker, _token=None))]
    pub fn branch_walker(&self, _walker: &Walker, _token: Option<String>) -> PyResult<Vec<Walker>> {
        Ok(vec![]) // Default empty implementation
    }

    // fn __eq__(&self, other: PyAny) -> PyResult<bool> {
    //     // Check if other is an Acceptor
    //     if !other.is_type_of::<Acceptor>()? {
    //         return Ok(false);
    //     }

    //     // Extract other's state_graph and compare
    //     let other: &Acceptor = other.extract()?;
    //     Ok(self.state_graph == other.state_graph)
    // }

    pub fn __str__(&self) -> String {
        format!("{}()", std::any::type_name::<Self>().split("::").last().unwrap_or("Acceptor"))
    }

    pub fn __repr__(&self) -> String {
        fn format_graph(graph: &StateGraph, indent: usize) -> String {
            if graph.is_empty() {
                return String::new();
            }

            let indent_str = "    ".repeat(indent);
            let mut lines = vec!["graph={\n".to_string()];

            for state in graph.keys().collect::<Vec<_>>() {
                if let Some(transitions) = graph.get(state) {
                    let transition_lines: Vec<String> = transitions
                        .iter()
                        .map(|(acceptor, target_state)| {
                            let acceptor_repr = format_acceptor(acceptor, indent + 2);
                            let target_str = match target_state {
                                State::Str(s) if s == "$" => "'$'".to_string(),
                                _ => format!("{:?}", target_state),
                            };
                            format!("({}, {})", acceptor_repr, target_str)
                        })
                        .collect();
                    lines.push(format!(
                        "{}    {:?}: [{}],\n",
                        indent_str,
                        state,
                        transition_lines.join(", ")
                    ));
                }
            }
            lines.push(format!("{}}}", indent_str));
            lines.join("")
        }

        fn format_acceptor(acceptor: &Acceptor, indent: usize) -> String {
            acceptor
                .__repr__()
                .lines()
                .enumerate()
                .map(|(idx, line)| {
                    if idx == 0 {
                        line.to_string()
                    } else {
                        format!("{}{}", "    ".repeat(indent), line)
                    }
                })
                .collect::<Vec<_>>()
                .join("\n")
        }

        let formatted_graph = format_graph(&self.state_graph, 0);
        format!(
            "{}({})",
            std::any::type_name::<Self>().split("::").last().unwrap_or("Acceptor"),
            formatted_graph
        )
    }

}
