//! Mycelic live demo: a visual console that drives the real stack, and a replay of a recorded run.
//!
//! ```text
//! mycelic-live                                   # start the stack, serve the console, open the browser
//! mycelic-live --record demo/live/trace.json     # headless: every act, write the trace
//! mycelic-live --replay                          # serve the recorded run, no stack needed
//! ```
//!
//! Everything is in `mycelic_harness::live`; its module documentation is the specification (acts, HTTP endpoints,
//! console contract, trace format, exit codes).

use clap::Parser;
use mycelic_harness::live::{Args, main as run};
use std::process::ExitCode;

fn main() -> ExitCode {
    ExitCode::from(run(Args::parse()))
}
