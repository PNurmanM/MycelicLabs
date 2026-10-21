//! Python semantics the outputs depend on: set iteration order, `json` text and format specs.

pub mod fmt;
pub mod json;
pub mod pyset;

pub use pyset::PySet;
