//! Mycelic: infrastructure for distributed agent memory.
//!
//! Agents keep knowledge locally; Mycelic lets the parts worth sharing propagate through an organization
//! (agent -> team -> department -> subsidiary -> region -> enterprise), aggregates them into higher-level memories
//! with full lineage, and makes them retrievable through an HTTP API, the SDK (`mycelic-sdk`) and MCP.
//!
//! See `DEPLOYMENT.md` for running it and `docs/MYCELIC_ARCHITECTURE.md` for what actually runs.

pub mod aggregation;
pub mod api;
pub mod auth;
pub mod config;
pub mod coordination;
pub mod hierarchy;
pub mod lineage;
pub mod mcp;
pub mod metrics;
pub mod models;
pub mod pyjson;
pub mod retrieval;
pub mod service;
pub mod store;
pub mod textutil;
pub mod transport;

pub const VERSION: &str = mycelic_sdk::VERSION;
