use pyo3::prelude::*;

mod engine;
mod state_machine;
mod walker;
mod acceptor;
#[pymodule]
fn pse_core(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<walker::Walker>()?;
    // m.add_class::<engine::StructuringEngine>()?;
    // m.add_class::<state_machine::StateMachine>()?;
    m.add_class::<acceptor::Acceptor>()?;
    Ok(())
}
