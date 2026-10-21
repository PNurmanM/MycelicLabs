//! Python semantics the outputs depend on: set iteration order, `json` text and format specs.

pub mod fmt;
pub mod json;
pub mod pyset;

pub use pyset::PySet;

/// Python's `x ** y` for floats: a runtime call into libm `pow`.
///
/// LLVM rewrites `powf` with a constant exponent (`x.powf(0.5)` becomes `sqrt`, `x.powf(2.0)` becomes `x * x`),
/// but CPython always calls `pow`, and Apple's `pow(x, 0.5)` differs from `sqrt(x)` in ~0.14% of inputs.  Hiding
/// the exponent from the optimiser keeps the call.
#[inline]
pub fn pow(x: f64, y: f64) -> f64 {
    x.powf(std::hint::black_box(y))
}
