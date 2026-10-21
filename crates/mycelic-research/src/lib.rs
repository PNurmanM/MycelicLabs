//! The Mycelic enterprise-hierarchy benchmark (`research/mycelic` in the Python era), ported to Rust.
//!
//! A synthetic six-level enterprise with hidden cross-organisational patterns, every architecture under test,
//! the metrics, the experiment suite E1-E12, the statistics and the generators of the published documents.
//! The port is held to the Python implementation (last Python commit `f7e954b`) bit for bit: `np` reproduces
//! numpy's random stream and reductions, `py` the Python behaviours the results depend on.

pub mod np;
pub mod py;
