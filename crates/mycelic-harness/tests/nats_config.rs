//! Port of tests/mycelic/test_nats_config.py.  The shipped broker config takes its credentials from the
//! environment, unquoted, and that constrains the password.
//!
//! `deploy/mycelic/nats/nats.conf` (and the Kubernetes ConfigMap) write `password: $NATS_PASSWORD`; nats-server
//! substitutes the value and then parses it as if it had been written there, so a password that starts with a digit
//! is read as a number (`1e5…` is a float) and the broker refuses to start.  Plain `openssl rand -hex 32` produces
//! such a value about one time in fifteen.  Quoting the reference is not a fix: nats-server then takes
//! `$NATS_PASSWORD` literally.  The generators start with a letter instead.
//!
//! These tests pin the generator's shape, that the documented commands generate a letter-first password, and that
//! the shipped config starts with a letter-first password whose hex part is the troublesome `1e5…` and admits a
//! client with it (and refuses a wrong one), i.e. the substitution still works.  The client is a raw NATS protocol
//! handshake, so the harness needs no NATS library.

mod support;

use mycelic_harness::{free_port, secret_hex};
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// `^[a-f][0-9a-f]{32}$`
fn matches_letter_first(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() == 33
        && (b'a'..=b'f').contains(&b[0])
        && b[1..].iter().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(c))
}

#[test]
fn test_secret_hex_always_starts_with_a_letter() {
    for _ in 0..5000 {
        assert!(matches_letter_first(&secret_hex(16)));
    }
}

#[test]
fn test_documented_generators_prefix_a_letter() {
    for rel in ["DEPLOYMENT.md", "deploy/mycelic/k8s/secret.example.yaml", "deploy/mycelic/.env.example"] {
        let text = std::fs::read_to_string(support::repo_root().join(rel)).unwrap();
        assert!(!text.contains("NATS_PASSWORD=$(openssl"), "{rel}: a bare openssl value can start with a digit");
        assert!(text.contains("n$(openssl rand -hex 32)"), "{rel}: the documented generator prefixes a letter");
    }
}

const PASSWORD: &str = "n1e50123456789abcdef0123456789abcdef"; // letter first; without the "n" this broke CI

struct Broker {
    child: Child,
    port: u16,
    _dir: tempfile::TempDir,
}

impl Drop for Broker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn start_shipped_config() -> Option<Broker> {
    let bin = mycelic_harness::nats_server_bin().filter(|p| p.is_file())?;
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let child = Command::new(bin)
        .arg("-c")
        .arg(support::repo_root().join("deploy/mycelic/nats/nats.conf"))
        .args(["-p", &port.to_string(), "-m", &free_port().to_string()])
        .env("NATS_USER", "mycelic")
        .env("NATS_PASSWORD", PASSWORD)
        .env("NATS_STORE_DIR", dir.path())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let broker = Broker { child, port, _dir: dir };
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        if TcpStream::connect(("127.0.0.1", port)).is_ok() {
            return Some(broker);
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    panic!("nats-server did not start with the shipped config and a letter-first password");
}

/// CONNECT with user/password and PING; true when the server answers PONG, false on an authorization error.
fn authenticates(port: u16, password: &str) -> bool {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
    stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    assert!(line.starts_with("INFO "), "server greeting: {line}");
    let connect =
        serde_json::json!({"verbose": false, "pedantic": false, "user": "mycelic", "pass": password, "protocol": 1});
    write!(stream, "CONNECT {connect}\r\nPING\r\n").unwrap();
    line.clear();
    reader.read_line(&mut line).unwrap();
    if line.starts_with("PONG") {
        return true;
    }
    assert!(line.contains("Authorization Violation"), "unexpected reply: {line}");
    false
}

#[test]
fn test_client_connects_with_the_substituted_password() {
    let Some(broker) = start_shipped_config() else {
        eprintln!("skipped: nats-server binary not available (set MYCELIC_NATS_SERVER_BIN)");
        return;
    };
    assert!(authenticates(broker.port, PASSWORD));
}

#[test]
fn test_wrong_password_is_refused() {
    let Some(broker) = start_shipped_config() else {
        eprintln!("skipped: nats-server binary not available (set MYCELIC_NATS_SERVER_BIN)");
        return;
    };
    assert!(!authenticates(broker.port, &format!("n{}", "f".repeat(32))));
}
