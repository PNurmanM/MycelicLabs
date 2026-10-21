//! Deployment drivers shared by the production smoke test and the demos.
//!
//! Both drive a *real* deployment (no mocks): [`ProcessDriver`] runs `nats-server` and `mycelic serve` as
//! subprocesses in a temporary directory (what CI uses); [`ComposeDriver`] drives `docker compose` against
//! `deploy/mycelic/docker-compose.yml` (what an operator runs).  Each exposes the same operations, so one scenario
//! can be executed against either:
//!
//! ```text
//! d.up()                        start everything, wait for /ready
//! d.kill("mycelic")             SIGKILL / docker compose kill
//! d.start("mycelic")            restart the service, wait for /ready
//! d.kill("nats"); d.start("nats")
//! d.wipe_database()             lose the Mycelic database (service down), keep the NATS volume
//! d.base_url(), d.admin_token(), d.logs("mycelic")
//! d.down()
//! ```
//!
//! This crate is agent side: it talks to a deployment over HTTP through `mycelic-sdk` and never links the service
//! (`tests/layout.rs` enforces it).  The service and agent processes are the `mycelic` and `mycelic-agent`
//! binaries, found next to the running executable, in `MYCELIC_BIN_DIR`, or on `PATH`.

pub mod live;
pub mod scenario;
pub mod smoke;
pub mod strategic;

use anyhow::{Context, Result, anyhow, bail};
use rand::Rng;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// The repository the drivers work in (rules file, compose file): `MYCELIC_REPO_ROOT`, or the checkout this crate
/// was built from.
pub fn repo_root() -> PathBuf {
    if let Some(p) = std::env::var_os("MYCELIC_REPO_ROOT") {
        return PathBuf::from(p);
    }
    let here = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    here.canonicalize().unwrap_or(here)
}

pub fn compose_file() -> PathBuf {
    repo_root().join("deploy/mycelic/docker-compose.yml")
}

pub fn rules_file() -> PathBuf {
    repo_root().join("deploy/mycelic/rules.json")
}

/// A hex secret that starts with a letter.  nats-server substitutes `$NATS_PASSWORD` into its config unquoted, so
/// a value that starts with a digit is read as a number (`1e5…` is a float) and the broker refuses to start.
pub fn secret_hex(nbytes: usize) -> String {
    let mut rng = rand::rng();
    let first = b"abcdef"[rng.random_range(0..6)] as char;
    let bytes: Vec<u8> = (0..nbytes).map(|_| rng.random()).collect();
    format!("{first}{}", hex::encode(bytes))
}

/// 32 random bytes as hex (`secrets.token_hex(32)`).
pub fn token_hex(nbytes: usize) -> String {
    let mut rng = rand::rng();
    hex::encode((0..nbytes).map(|_| rng.random::<u8>()).collect::<Vec<_>>())
}

pub fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0").and_then(|l| l.local_addr()).map(|a| a.port()).unwrap_or(0)
}

/// Poll `url` until it answers `status`; the body as JSON (empty object when there is none).
pub fn wait_http(url: &str, timeout: Duration, status: u16, headers: &[(&str, &str)]) -> Result<Value> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(3)))
        .http_status_as_error(false)
        .build()
        .into();
    let deadline = Instant::now() + timeout;
    let mut last = String::from("no answer");
    while Instant::now() < deadline {
        let mut req = agent.get(url);
        for (k, v) in headers {
            req = req.header(*k, *v);
        }
        match req.call() {
            Ok(mut r) if r.status().as_u16() == status => {
                let raw = r.body_mut().read_to_vec().unwrap_or_default();
                return Ok(if raw.is_empty() {
                    Value::Object(Default::default())
                } else {
                    serde_json::from_slice(&raw).unwrap_or(Value::Null)
                });
            }
            Ok(r) => last = format!("HTTP {}", r.status().as_u16()),
            Err(e) => last = e.to_string(),
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    bail!("{url} did not answer {status} within {}s (last: {last})", timeout.as_secs())
}

/// Where a binary of this workspace lives: `MYCELIC_BIN_DIR`, the directory of the running executable (or one of its
/// parents: `target/debug/deps` for tests), the repository's `target/{release,debug}`, or `PATH`.
pub fn binary(name: &str) -> PathBuf {
    if let Some(d) = std::env::var_os("MYCELIC_BIN_DIR") {
        return PathBuf::from(d).join(name);
    }
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        candidates.extend(exe.ancestors().skip(1).take(3).map(|d| d.join(name)));
    }
    let target = repo_root().join("target");
    candidates.extend(["release", "debug"].iter().map(|p| target.join(p).join(name)));
    if let Some(paths) = std::env::var_os("PATH") {
        candidates.extend(std::env::split_paths(&paths).map(|d| d.join(name)));
    }
    candidates.into_iter().find(|p| p.is_file()).unwrap_or_else(|| PathBuf::from(name))
}

/// `nats-server` from `MYCELIC_NATS_SERVER_BIN` or PATH.
pub fn nats_server_bin() -> Option<PathBuf> {
    match std::env::var("MYCELIC_NATS_SERVER_BIN") {
        Ok(p) if !p.is_empty() => Some(PathBuf::from(p)),
        _ => std::env::var_os("PATH")
            .and_then(|paths| std::env::split_paths(&paths).map(|d| d.join("nats-server")).find(|p| p.is_file())),
    }
}

/// One deployment, driven the same way whether it runs as local processes or under docker compose.
pub trait Driver {
    fn name(&self) -> &'static str;
    fn base_url(&self) -> String;
    fn admin_token(&self) -> String;
    fn up(&mut self) -> Result<()>;
    fn down(&mut self);
    fn kill(&mut self, service: &str) -> Result<()>;
    fn stop(&mut self, service: &str) -> Result<()>;
    fn start(&mut self, service: &str) -> Result<()>;
    fn wipe_database(&mut self) -> Result<()>;
    fn logs(&self, service: &str, tail: usize) -> String;
    /// Secrets a recording must never contain (the live demo's trace guard).
    fn secrets(&self) -> Vec<String> {
        vec![self.admin_token()]
    }
    /// The process ids this driver started (the live demo checks none outlives the run).
    fn pids(&self) -> Vec<u32> {
        Vec::new()
    }

    fn wait_ready(&self, timeout: Duration) -> Result<()> {
        wait_http(&format!("{}/ready", self.base_url()), timeout, 200, &[]).map(|_| ())
    }

    fn wait_health(&self, timeout: Duration) -> Result<Value> {
        wait_http(&format!("{}/health", self.base_url()), timeout, 200, &[])
    }
}

/// nats-server + mycelic as local subprocesses (needs the nats-server binary).
pub struct ProcessDriver {
    pub workdir: PathBuf,
    _tmp: Option<tempfile::TempDir>,
    pub nats_bin: PathBuf,
    pub mycelic_bin: PathBuf,
    pub nats_port: u16,
    pub http_port: u16,
    pub admin_token: String,
    pub signing_key: String,
    pub nats_user: String,
    pub nats_password: String,
    pub min_support: u32,
    pub log_level: String,
    pub rules_file: PathBuf,
    procs: HashMap<String, Child>,
}

impl ProcessDriver {
    pub fn new() -> Result<ProcessDriver> {
        Self::in_dir(None)
    }

    pub fn in_dir(workdir: Option<PathBuf>) -> Result<ProcessDriver> {
        let nats_bin = nats_server_bin()
            .ok_or_else(|| anyhow!("nats-server not found: set MYCELIC_NATS_SERVER_BIN or put nats-server on PATH"))?;
        let (tmp, workdir) = match workdir {
            Some(w) => (None, w),
            None => {
                let t = tempfile::Builder::new().prefix("mycelic-smoke-").tempdir()?;
                let p = t.path().to_path_buf();
                (Some(t), p)
            }
        };
        std::fs::create_dir_all(workdir.join("js"))?;
        let d = ProcessDriver {
            nats_bin,
            mycelic_bin: binary("mycelic"),
            nats_port: free_port(),
            http_port: free_port(),
            admin_token: token_hex(32),
            signing_key: token_hex(32),
            nats_user: "mycelic".into(),
            nats_password: secret_hex(16),
            min_support: 2,
            log_level: "INFO".into(),
            rules_file: rules_file(),
            procs: HashMap::new(),
            workdir,
            _tmp: tmp,
        };
        std::fs::write(
            d.workdir.join("nats.conf"),
            format!(
                "port: {}\nhttp_port: -1\njetstream {{ store_dir: \"{}\" }}\nauthorization {{ user: \"{}\", password: \"{}\" }}\n",
                d.nats_port,
                d.workdir.join("js").display(),
                d.nats_user,
                d.nats_password
            ),
        )?;
        Ok(d)
    }

    /// The service's environment: the caller's, without any MYCELIC_* it set, plus this deployment's.
    pub fn env(&self) -> Vec<(String, String)> {
        let mut env: Vec<(String, String)> = std::env::vars().filter(|(k, _)| !k.starts_with("MYCELIC_")).collect();
        env.extend(
            [
                ("MYCELIC_HOST", "127.0.0.1".to_string()),
                ("MYCELIC_PORT", self.http_port.to_string()),
                ("MYCELIC_DB_PATH", self.workdir.join("mycelic.db").to_string_lossy().into_owned()),
                ("MYCELIC_NATS_URL", format!("nats://127.0.0.1:{}", self.nats_port)),
                ("MYCELIC_NATS_USER", self.nats_user.clone()),
                ("MYCELIC_NATS_PASSWORD", self.nats_password.clone()),
                ("MYCELIC_ADMIN_TOKEN", self.admin_token.clone()),
                ("MYCELIC_EVENT_SIGNING_KEY", self.signing_key.clone()),
                ("MYCELIC_RULES_FILE", self.rules_file.to_string_lossy().into_owned()),
                ("MYCELIC_MIN_SUPPORT", self.min_support.to_string()),
                ("MYCELIC_LOG_LEVEL", self.log_level.clone()),
                ("MYCELIC_RATE_LIMIT_RPS", "1000".to_string()),
                ("MYCELIC_RATE_LIMIT_BURST", "2000".to_string()),
            ]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v)),
        );
        env
    }

    fn log_file(&self, service: &str) -> Result<File> {
        Ok(OpenOptions::new().create(true).append(true).open(self.workdir.join(format!("{service}.log")))?)
    }
}

impl Driver for ProcessDriver {
    fn name(&self) -> &'static str {
        "process"
    }

    fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.http_port)
    }

    fn admin_token(&self) -> String {
        self.admin_token.clone()
    }

    fn secrets(&self) -> Vec<String> {
        vec![self.admin_token.clone(), self.signing_key.clone(), self.nats_password.clone()]
    }

    fn pids(&self) -> Vec<u32> {
        self.procs.values().map(Child::id).collect()
    }

    fn up(&mut self) -> Result<()> {
        self.start("nats")?;
        self.start("mycelic")
    }

    fn down(&mut self) {
        let names: Vec<String> = self.procs.keys().cloned().collect();
        for name in names {
            let _ = self.kill(&name);
        }
    }

    fn kill(&mut self, service: &str) -> Result<()> {
        if let Some(mut p) = self.procs.remove(service) {
            // SIGKILL: no graceful shutdown, exactly like a crashed pod
            let _ = p.kill();
            let _ = p.wait();
        }
        Ok(())
    }

    fn stop(&mut self, service: &str) -> Result<()> {
        if let Some(mut p) = self.procs.remove(service) {
            let _ = Command::new("kill").arg("-TERM").arg(p.id().to_string()).status();
            let deadline = Instant::now() + Duration::from_secs(20);
            while Instant::now() < deadline {
                if p.try_wait()?.is_some() {
                    return Ok(());
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            let _ = p.kill();
            let _ = p.wait();
        }
        Ok(())
    }

    fn start(&mut self, service: &str) -> Result<()> {
        if self.procs.contains_key(service) {
            return Ok(());
        }
        let log = self.log_file(service)?;
        match service {
            "nats" => {
                let child = Command::new(&self.nats_bin)
                    .arg("-c")
                    .arg(self.workdir.join("nats.conf"))
                    .stdout(Stdio::from(log.try_clone()?))
                    .stderr(Stdio::from(log))
                    .spawn()
                    .with_context(|| format!("starting {}", self.nats_bin.display()))?;
                self.procs.insert("nats".into(), child);
                let deadline = Instant::now() + Duration::from_secs(15);
                while Instant::now() < deadline {
                    if let Some(status) = self.procs.get_mut("nats").and_then(|c| c.try_wait().ok().flatten()) {
                        self.procs.remove("nats");
                        bail!("nats-server did not start (exit={status}):\n{}", self.logs("nats", 15));
                    }
                    if TcpStream::connect(("127.0.0.1", self.nats_port)).is_ok() {
                        return Ok(());
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
                bail!("nats-server did not start:\n{}", self.logs("nats", 15))
            }
            "mycelic" => {
                let child = Command::new(&self.mycelic_bin)
                    .arg("serve")
                    .env_clear()
                    .envs(self.env())
                    .current_dir(repo_root())
                    .stdout(Stdio::from(log.try_clone()?))
                    .stderr(Stdio::from(log))
                    .spawn()
                    .with_context(|| format!("starting {}", self.mycelic_bin.display()))?;
                self.procs.insert("mycelic".into(), child);
                self.wait_ready(Duration::from_secs(60))
            }
            other => bail!("unknown service {other}"),
        }
    }

    fn wipe_database(&mut self) -> Result<()> {
        if self.procs.contains_key("mycelic") {
            bail!("stop the service before wiping its database");
        }
        for entry in std::fs::read_dir(&self.workdir)? {
            let p = entry?.path();
            if p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("mycelic.db")) {
                std::fs::remove_file(p)?;
            }
        }
        Ok(())
    }

    fn logs(&self, service: &str, tail: usize) -> String {
        let text = std::fs::read_to_string(self.workdir.join(format!("{service}.log"))).unwrap_or_default();
        let lines: Vec<&str> = text.lines().collect();
        lines[lines.len().saturating_sub(tail)..].join("\n")
    }
}

impl Drop for ProcessDriver {
    fn drop(&mut self) {
        self.down();
    }
}

/// docker compose against deploy/mycelic/docker-compose.yml, with a generated .env unless one is given.
pub struct ComposeDriver {
    pub compose_file: PathBuf,
    pub project: String,
    pub build: bool,
    generated_env: bool,
    pub env_file: PathBuf,
    pub port: u16,
    pub admin_token: String,
}

impl ComposeDriver {
    pub fn new(
        env_file: Option<PathBuf>,
        project: &str,
        build: bool,
        extra_env: &[(String, String)],
    ) -> Result<ComposeDriver> {
        let compose = compose_file();
        let generated_env = env_file.is_none();
        let env_file =
            env_file.unwrap_or_else(|| compose.parent().unwrap_or(Path::new(".")).join(format!(".env.{project}")));
        let (port, admin_token) = if generated_env {
            let port = free_port();
            let token = token_hex(32);
            let mut lines = vec![
                format!("MYCELIC_ADMIN_TOKEN={token}"),
                format!("MYCELIC_EVENT_SIGNING_KEY={}", token_hex(32)),
                "NATS_USER=mycelic".to_string(),
                format!("NATS_PASSWORD={}", secret_hex(16)),
                format!("MYCELIC_PORT={port}"),
                "MYCELIC_LOG_LEVEL=INFO".to_string(),
            ];
            lines.extend(extra_env.iter().map(|(k, v)| format!("{k}={v}")));
            std::fs::write(&env_file, lines.join("\n") + "\n")?;
            (port, token)
        } else {
            let text = std::fs::read_to_string(&env_file)?;
            let values: HashMap<&str, &str> = text
                .lines()
                .filter(|l| l.contains('=') && !l.starts_with('#'))
                .filter_map(|l| l.split_once('='))
                .collect();
            let token = values
                .get("MYCELIC_ADMIN_TOKEN")
                .map(|s| s.trim().to_string())
                .ok_or_else(|| anyhow!("{} lacks MYCELIC_ADMIN_TOKEN", env_file.display()))?;
            (values.get("MYCELIC_PORT").and_then(|p| p.trim().parse().ok()).unwrap_or(8080), token)
        };
        Ok(ComposeDriver {
            compose_file: compose,
            project: project.to_string(),
            build,
            generated_env,
            env_file,
            port,
            admin_token,
        })
    }

    fn compose(&self, args: &[&str], check: bool, capture: bool) -> Result<String> {
        let mut cmd = Command::new("docker");
        cmd.args(["compose", "-p", &self.project, "--env-file"])
            .arg(&self.env_file)
            .arg("-f")
            .arg(&self.compose_file)
            .args(args);
        if capture {
            let out = cmd.output()?;
            if check && !out.status.success() {
                bail!("docker compose {} failed: {}", args.join(" "), String::from_utf8_lossy(&out.stderr));
            }
            return Ok(String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr));
        }
        let status = cmd.status()?;
        if check && !status.success() {
            bail!("docker compose {} failed ({status})", args.join(" "));
        }
        Ok(String::new())
    }
}

impl Driver for ComposeDriver {
    fn name(&self) -> &'static str {
        "compose"
    }

    fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    fn admin_token(&self) -> String {
        self.admin_token.clone()
    }

    fn up(&mut self) -> Result<()> {
        // a smoke run means a fresh deployment: whatever a previous run of this project left behind goes first
        let _ = self.compose(&["down", "-v", "--remove-orphans"], false, true);
        let mut args = vec!["up", "-d", "--wait", "--wait-timeout", "180"];
        if self.build {
            args.insert(1, "--build");
        }
        self.compose(&args, true, false)?;
        self.wait_ready(Duration::from_secs(120))
    }

    fn down(&mut self) {
        let _ = self.compose(&["down", "-v", "--remove-orphans"], false, false);
        if self.generated_env {
            let _ = std::fs::remove_file(&self.env_file);
        }
    }

    fn kill(&mut self, service: &str) -> Result<()> {
        self.compose(&["kill", service], true, false).map(|_| ())
    }

    fn stop(&mut self, service: &str) -> Result<()> {
        self.compose(&["stop", service], true, false).map(|_| ())
    }

    fn start(&mut self, service: &str) -> Result<()> {
        self.compose(&["start", service], true, false)?;
        if service == "mycelic" {
            self.wait_ready(Duration::from_secs(120))
        } else {
            std::thread::sleep(Duration::from_secs(1));
            Ok(())
        }
    }

    fn wipe_database(&mut self) -> Result<()> {
        // the service must be stopped; delete the SQLite files inside the named volume, keep the NATS volume
        let volume = format!("{}_mycelic-data:/data", self.project);
        let status = Command::new("docker")
            .args([
                "run",
                "--rm",
                "--user",
                "root",
                "--entrypoint",
                "sh",
                "-v",
                &volume,
                "mycelic:local",
                "-c",
                "rm -f /data/mycelic.db*",
            ])
            .status()?;
        if !status.success() {
            bail!("wiping the database volume failed ({status})");
        }
        Ok(())
    }

    fn logs(&self, service: &str, tail: usize) -> String {
        self.compose(&["logs", "--no-color", "--tail", &tail.to_string(), service], false, true).unwrap_or_default()
    }
}

/// `--driver process|compose` for the smoke test and the demos.
pub fn make_driver(kind: &str, project: &str, env_file: Option<PathBuf>, build: bool) -> Result<Box<dyn Driver>> {
    match kind {
        "process" => Ok(Box::new(ProcessDriver::new()?)),
        "compose" => {
            let extra: Vec<(String, String)> = std::env::var("MYCELIC_BUILD_CA_BUNDLE")
                .ok()
                .filter(|v| !v.is_empty())
                .map(|v| vec![("CA_BUNDLE_FILE".to_string(), v)])
                .unwrap_or_default();
            Ok(Box::new(ComposeDriver::new(env_file, project, build, &extra)?))
        }
        other => bail!("unknown driver {other}"),
    }
}

/// Collapse whitespace and cut to `n` characters with an ellipsis (the demos' `clip`).
pub fn clip(t: &str, n: usize) -> String {
    let t = t.split_whitespace().collect::<Vec<_>>().join(" ");
    if t.chars().count() <= n { t } else { format!("{}…", t.chars().take(n.saturating_sub(1)).collect::<String>()) }
}

/// The last segment of a unit path.
pub fn leaf(scope: &str) -> &str {
    scope.rsplit('/').next().unwrap_or(scope)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secret_hex_always_starts_with_a_letter() {
        for _ in 0..5000 {
            let s = secret_hex(16);
            assert_eq!(s.len(), 33);
            assert!(s.as_bytes()[0].is_ascii_lowercase() && b"abcdef".contains(&s.as_bytes()[0]));
            assert!(s.bytes().all(|b| b.is_ascii_hexdigit()));
        }
    }
}
