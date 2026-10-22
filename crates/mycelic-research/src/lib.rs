//! The Mycelic enterprise-hierarchy benchmark (`research/mycelic` in the Python era), ported to Rust.
//!
//! A synthetic six-level enterprise with hidden cross-organisational patterns, every architecture under test,
//! the metrics, the experiment suite E1-E12, the statistics and the generators of the published documents.
//! The port is held to the Python implementation (last Python commit `f7e954b`) bit for bit: `np` reproduces
//! numpy's random stream and reductions, `py` the Python behaviours the results depend on.

// The Python's records are tuples and nested dicts; spelling each as a type alias would hide the correspondence
// with the original rather than clarify it.
#![allow(clippy::type_complexity)]

pub mod analysis;
pub mod calibrator;
pub mod corpus;
pub mod evalm;
pub mod experiments;
pub mod loss_accounting;
pub mod loss_doc;
pub mod loss_report;
pub mod models;
pub mod np;
pub mod ops;
pub mod org;
pub mod py;
pub mod runner;
pub mod systems;
pub mod vnext_data;
