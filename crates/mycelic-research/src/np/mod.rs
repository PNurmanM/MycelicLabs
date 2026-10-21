//! numpy, as far as the benchmark's results depend on it: the random stream and the float reductions.

pub mod math;
pub mod random;
pub mod ziggurat;

pub use random::Generator;
