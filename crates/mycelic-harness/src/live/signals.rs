//! Ctrl-C, SIGTERM and SIGHUP stop the demo cooperatively.
//!
//! The Python demo turned every stop signal into a `KeyboardInterrupt` raised wherever the main thread happened to
//! be.  Rust has no asynchronous exceptions, so the handlers (installed with `signal-hook`) only set a flag, and
//! every blocking loop of the engine polls it: [`checkpoint`] between steps, the agent runner's `interrupted`
//! callback, [`sleep`] and [`wait_or_interrupt`].  Because a handler never terminates the process, a second signal
//! cannot cut the cleanup short; [`ignoring`] additionally forgets signals that arrive during a cleanup section,
//! like the `SIG_IGN` the Python version installed there.

use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// The engine was asked to stop (Ctrl-C, SIGTERM, SIGHUP).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Interrupted;

impl std::fmt::Display for Interrupted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("interrupted")
    }
}

impl std::error::Error for Interrupted {}

fn flag() -> &'static Arc<AtomicBool> {
    static FLAG: OnceLock<Arc<AtomicBool>> = OnceLock::new();
    FLAG.get_or_init(|| Arc::new(AtomicBool::new(false)))
}

/// Route SIGINT, SIGTERM and SIGHUP (the terminal window closed, an ssh session dropped) to the stop flag.
pub fn install() -> std::io::Result<()> {
    #[cfg(unix)]
    let signals = [signal_hook::consts::SIGINT, signal_hook::consts::SIGTERM, signal_hook::consts::SIGHUP];
    #[cfg(not(unix))]
    let signals = [signal_hook::consts::SIGINT, signal_hook::consts::SIGTERM];
    for sig in signals {
        signal_hook::flag::register(sig, Arc::clone(flag()))?;
    }
    Ok(())
}

pub fn interrupted() -> bool {
    flag().load(Ordering::SeqCst)
}

/// `Err(Interrupted)` once a stop signal arrived: the cooperative stand-in for a raised `KeyboardInterrupt`.
pub fn checkpoint() -> Result<(), Interrupted> {
    if interrupted() { Err(Interrupted) } else { Ok(()) }
}

/// Run cleanup that a stop signal must not change: one that arrives meanwhile is forgotten afterwards.
pub fn ignoring<T>(f: impl FnOnce() -> T) -> T {
    let before = interrupted();
    let out = f();
    if !before {
        flag().store(false, Ordering::SeqCst);
    }
    out
}

/// Sleep, waking early (with `Err`) when a stop signal arrives.
pub fn sleep(d: Duration) -> Result<(), Interrupted> {
    let deadline = Instant::now() + d;
    loop {
        checkpoint()?;
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Ok(());
        }
        std::thread::sleep(left.min(Duration::from_millis(50)));
    }
}

/// Wait `seconds` (forever with `None`) or until a stop signal arrives; `Err` means a signal ended the wait.
pub fn wait_or_interrupt(seconds: Option<f64>) -> Result<(), Interrupted> {
    match seconds {
        Some(s) => sleep(Duration::from_secs_f64(s.max(0.0))),
        None => loop {
            sleep(Duration::from_millis(250))?;
        },
    }
}
