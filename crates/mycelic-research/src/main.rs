//! `mycelic-research <module> [args...]`: one subcommand per module of the Python research code, taking the same
//! arguments the module's `python3 -m research.mycelic.<module>` took.

use anyhow::{Result, bail};
use mycelic_research::{live_rank, live_tasks, plots, score_live};

type Entry = fn(&[String]) -> Result<()>;

/// (subcommand, what it does, entry point)
fn modules() -> Vec<(&'static str, &'static str, Entry)> {
    vec![
        ("live_tasks", "build the operator test-set and its answer key (seed 900, 2,000 users)", live_tasks::main),
        ("score_live", "score the live operator answers and fit q to each model", score_live::main),
        (
            "live_rank",
            "candidate-discrimination tasks: build, or `score` the answers; `rich` adds source notes",
            live_rank::main,
        ),
        (
            "plots",
            "the report figures from the rows; optional output directory (default artifacts/figures)",
            plots::main,
        ),
    ]
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mods = modules();
    let Some(cmd) = args.get(1) else {
        eprintln!("usage: mycelic-research <module> [args...]\n\nmodules:");
        for (name, what, _) in &mods {
            eprintln!("  {name:16} {what}");
        }
        std::process::exit(2);
    };
    match mods.iter().find(|m| m.0 == cmd.as_str()) {
        Some((_, _, entry)) => entry(&args[2..]),
        None => bail!("unknown module {cmd:?}; run without arguments for the list"),
    }
}
