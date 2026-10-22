//! `mycelic-research <module> [args...]`: one subcommand per module of the Python research code, taking the same
//! arguments the module's `python3 -m research.mycelic.<module>` took.

use anyhow::{Result, bail};

type Entry = fn(&[String]) -> Result<()>;

/// (subcommand, what it does, entry point)
fn modules() -> Vec<(&'static str, &'static str, Entry)> {
    vec![
        ("experiments", "E1-E12, each appending rows to its own JSONL", mycelic_research::experiments::main),
        ("loss_accounting", "per-pattern loss funnel into loss_funnel.jsonl", mycelic_research::loss_accounting::main),
        (
            "quick_paired",
            "paired A/B of a variant against a base on identical worlds",
            mycelic_research::quick_paired::main,
        ),
        ("arm_paired", "multi-arm paired runs, each arm with its own ranker", mycelic_research::arm_paired::main),
        ("calibrate", "fit the v1 knobs on the calibration seeds", mycelic_research::calibrate::main),
        (
            "calibrator",
            "the learned ranker: dump, select, store, archs, final, fit",
            mycelic_research::calibrator::main,
        ),
        ("final_rerun", "final_rerun.sh's calibration-file steps", mycelic_research::final_rerun::main),
        (
            "freeze_vnext",
            "freeze vNext knobs into calibration.json with provenance",
            mycelic_research::freeze_vnext::main,
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
