//! Production smoke test for Mycelic: the canonical deployment demonstration.
//!
//! ```text
//! mycelic-smoke --driver process            # nats-server + service as local processes
//! mycelic-smoke --driver compose            # docker compose (builds the image)
//! mycelic-smoke --driver compose --env-file deploy/mycelic/.env --no-build   # a running stack's config
//! ```
//!
//! What it proves, in order, is listed in `mycelic_harness::smoke`; the run fails on the first broken step.  Exit
//! code 0 means every assertion held, 1 that one failed, 2 that the test itself could not run.  A JSON report is
//! written with `--report`.

use clap::Parser;
use mycelic_harness::smoke::{SmokeFailure, run};
use mycelic_harness::{Driver, make_driver};
use serde_json::json;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser)]
#[command(
    name = "mycelic-smoke",
    about = "Production smoke test for Mycelic (nine asserted steps against a real deployment)"
)]
struct Args {
    #[arg(long, default_value = "process", value_parser = ["process", "compose"])]
    driver: String,
    /// 3 teams; 2 -> 6 agents, 33 -> 99 agents
    #[arg(long, default_value_t = 2)]
    agents_per_team: usize,
    /// compose: use this .env (its MYCELIC_ADMIN_TOKEN/MYCELIC_PORT)
    #[arg(long)]
    env_file: Option<PathBuf>,
    /// compose: do not rebuild the image
    #[arg(long)]
    no_build: bool,
    /// compose project name
    #[arg(long, default_value = "mycelic-smoke")]
    project: String,
    /// write a JSON report here
    #[arg(long)]
    report: Option<PathBuf>,
    /// keep the deployment running after a pass
    #[arg(long)]
    keep: bool,
}

fn main() -> ExitCode {
    let args = Args::parse();
    let log = |m: &str| println!("{m}");
    let mut driver: Box<dyn Driver> =
        match make_driver(&args.driver, &args.project, args.env_file.clone(), !args.no_build) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("ERROR: {e:#}");
                return ExitCode::from(2);
            }
        };
    let workdir = tempfile::Builder::new().prefix("mycelic-agents-").tempdir().expect("temporary directory");
    let (code, report) = match run(driver.as_mut(), args.agents_per_team, workdir.path(), &log) {
        Ok(report) => {
            let text: String = report["final_answer"]["text"].as_str().unwrap_or("").chars().take(160).collect();
            log(&format!("\nPASS in {}s: {text}", report["seconds"]));
            (0u8, report)
        }
        Err(e) if e.downcast_ref::<SmokeFailure>().is_some() => {
            log(&format!("\nFAIL: {e}"));
            log(&format!("--- mycelic log tail ---\n{}", driver.logs("mycelic", 40)));
            log(&format!("--- nats log tail ---\n{}", driver.logs("nats", 20)));
            (1, json!({"result": "FAIL", "error": e.to_string()}))
        }
        Err(e) => {
            log(&format!("\nERROR: {e:#}"));
            log(&format!("--- mycelic log tail ---\n{}", driver.logs("mycelic", 60)));
            (2, json!({"result": "ERROR", "error": format!("{e:#}")}))
        }
    };
    if let Some(path) = &args.report {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(path, serde_json::to_string_pretty(&report).unwrap_or_default());
    }
    if args.keep && code == 0 {
        println!("deployment left running at {}  (MYCELIC_ADMIN_TOKEN={})", driver.base_url(), driver.admin_token());
        // the processes outlive this one, as they did with the Python harness
        std::mem::forget(driver);
        std::mem::forget(workdir);
    } else {
        driver.down();
    }
    ExitCode::from(code)
}
