//! `numpy.random.default_rng(seed)`, bit for bit.
//!
//! Every number the benchmark publishes is a function of the draws a numpy `Generator` made, so this module
//! reproduces numpy 2.4's stream exactly rather than approximating its distributions: `SeedSequence` entropy
//! mixing, the PCG64 (XSL-RR 128/64) bit generator including its half-used 32-bit buffer, and each `Generator`
//! method the simulator calls, translated from numpy's C and Cython sources (`distributions.c`,
//! `_bounded_integers.pyx.in`, `_generator.pyx`).  `tests/numpy_rng.rs` replays draws recorded from numpy itself.
//!
//! Two details are easy to get wrong and both change every later draw:
//! * a 32-bit draw takes the low half of a fresh 64-bit output and keeps the high half for the next 32-bit draw
//!   (`has_uint32`), while 64-bit draws and doubles never touch that buffer;
//! * bounded integers whose range fits in 32 bits use the 32-bit Lemire path, wider ranges the 64-bit one, and a
//!   range of one value consumes nothing.
//!
//! Floating-point contraction.  numpy's C is compiled with the compiler's default `-ffp-contract`, so on arm64
//! (where the reference numpy for these tests runs) `a * b + c` inside one expression becomes a fused multiply-add
//! in the normal sampler, the ziggurat's rejection test and the binomial samplers.  The fused operations below
//! (`mul_add`) were read off the disassembly of numpy 2.4.6's `_generator` module for macOS arm64, instruction by
//! instruction.  An x86-64 build without FMA rounds those intermediates separately; the difference is in the last
//! bit of intermediate values, and it changed none of the committed result rows (`tests/golden/`).

use super::ziggurat::{FI_DOUBLE, KI_DOUBLE, WI_DOUBLE, ZIGGURAT_NOR_INV_R, ZIGGURAT_NOR_R};

const PCG_MULT: u128 = (2549297995355413924u128 << 64) | 4865540595714422341u128;

const INIT_A: u32 = 0x43b0d7e5;
const MULT_A: u32 = 0x931e8875;
const INIT_B: u32 = 0x8b51f9dd;
const MULT_B: u32 = 0x58f38ded;
const MIX_MULT_L: u32 = 0xca01f9dd;
const MIX_MULT_R: u32 = 0x4973f715;
const XSHIFT: u32 = 16;
const POOL_SIZE: usize = 4;

fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
    let mut value = value ^ *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    value = value.wrapping_mul(*hash_const);
    value ^ (value >> XSHIFT)
}

fn mix(x: u32, y: u32) -> u32 {
    let result = MIX_MULT_L.wrapping_mul(x).wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    result ^ (result >> XSHIFT)
}

/// `SeedSequence(seed).generate_state(n_words, np.uint32)` for a non-negative integer seed (no spawn key).
pub fn seed_sequence_state(seed: u128, n_words: usize) -> Vec<u32> {
    // _int_to_uint32_array: little-endian 32-bit words, [0] for zero
    let mut entropy: Vec<u32> = Vec::new();
    let mut n = seed;
    if n == 0 {
        entropy.push(0);
    }
    while n > 0 {
        entropy.push((n & 0xFFFF_FFFF) as u32);
        n >>= 32;
    }
    let mut pool = [0u32; POOL_SIZE];
    let mut hash_const = INIT_A;
    for (i, slot) in pool.iter_mut().enumerate() {
        let word = entropy.get(i).copied().unwrap_or(0);
        *slot = hashmix(word, &mut hash_const);
    }
    for i_src in 0..POOL_SIZE {
        for i_dst in 0..POOL_SIZE {
            if i_src != i_dst {
                pool[i_dst] = mix(pool[i_dst], hashmix(pool[i_src], &mut hash_const));
            }
        }
    }
    for &word in entropy.iter().skip(POOL_SIZE) {
        for slot in pool.iter_mut() {
            *slot = mix(*slot, hashmix(word, &mut hash_const));
        }
    }
    let mut hash_const = INIT_B;
    let mut out = Vec::with_capacity(n_words);
    for i in 0..n_words {
        let mut data = pool[i % POOL_SIZE];
        data ^= hash_const;
        hash_const = hash_const.wrapping_mul(MULT_B);
        data = data.wrapping_mul(hash_const);
        data ^= data >> XSHIFT;
        out.push(data);
    }
    out
}

/// The PCG64 bit generator with numpy's 32-bit buffer.
#[derive(Clone, Debug)]
pub struct Pcg64 {
    state: u128,
    inc: u128,
    has_uint32: bool,
    uinteger: u32,
}

impl Pcg64 {
    pub fn from_seed(seed: u128) -> Self {
        let words = seed_sequence_state(seed, 8);
        let w64 = |i: usize| (words[2 * i] as u64) | ((words[2 * i + 1] as u64) << 32);
        let initstate = ((w64(0) as u128) << 64) | w64(1) as u128;
        let initseq = ((w64(2) as u128) << 64) | w64(3) as u128;
        let mut rng = Pcg64 { state: 0, inc: (initseq << 1) | 1, has_uint32: false, uinteger: 0 };
        rng.step();
        rng.state = rng.state.wrapping_add(initstate);
        rng.step();
        rng
    }

    #[inline]
    fn step(&mut self) {
        self.state = self.state.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let hi = (self.state >> 64) as u64;
        let lo = self.state as u64;
        (hi ^ lo).rotate_right((self.state >> 122) as u32)
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        if self.has_uint32 {
            self.has_uint32 = false;
            return self.uinteger;
        }
        let next = self.next_u64();
        self.has_uint32 = true;
        self.uinteger = (next >> 32) as u32;
        next as u32
    }

    #[inline]
    pub fn next_double(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }
}

/// `numpy.random.Generator` over PCG64: the methods the benchmark calls, with numpy's argument conventions.
#[derive(Clone, Debug)]
pub struct Generator {
    bits: Pcg64,
}

impl Generator {
    /// `np.random.default_rng(seed)`
    pub fn new(seed: u64) -> Self {
        Generator { bits: Pcg64::from_seed(seed as u128) }
    }

    pub fn bits(&mut self) -> &mut Pcg64 {
        &mut self.bits
    }

    // ---- bounded integers ------------------------------------------------------------------------------------

    fn lemire32(&mut self, rng: u32) -> u32 {
        let rng_excl = rng.wrapping_add(1);
        let mut m = self.bits.next_u32() as u64 * rng_excl as u64;
        let mut leftover = m as u32;
        if leftover < rng_excl {
            let threshold = (u32::MAX - rng) % rng_excl;
            while leftover < threshold {
                m = self.bits.next_u32() as u64 * rng_excl as u64;
                leftover = m as u32;
            }
        }
        (m >> 32) as u32
    }

    fn lemire64(&mut self, rng: u64) -> u64 {
        let rng_excl = rng.wrapping_add(1);
        let mut m = self.bits.next_u64() as u128 * rng_excl as u128;
        let mut leftover = m as u64;
        if leftover < rng_excl {
            let threshold = (u64::MAX - rng) % rng_excl;
            while leftover < threshold {
                m = self.bits.next_u64() as u128 * rng_excl as u128;
                leftover = m as u64;
            }
        }
        (m >> 64) as u64
    }

    /// `random_bounded_uint64(off=0, rng, use_masked=False)`: a value in `[0, rng]`.
    pub fn bounded_u64(&mut self, rng: u64) -> u64 {
        if rng == 0 {
            0
        } else if rng <= 0xFFFF_FFFF {
            if rng == 0xFFFF_FFFF { self.bits.next_u32() as u64 } else { self.lemire32(rng as u32) as u64 }
        } else if rng == u64::MAX {
            self.bits.next_u64()
        } else {
            self.lemire64(rng)
        }
    }

    /// `rng.integers(low, high)` (int64, `high` exclusive).
    pub fn integers(&mut self, low: i64, high: i64) -> i64 {
        assert!(low < high, "integers: low >= high ({low} >= {high})");
        let rng = (high - 1).wrapping_sub(low) as u64;
        low.wrapping_add(self.bounded_u64(rng) as i64)
    }

    /// `rng.integers(low, high, endpoint=True)`
    pub fn integers_closed(&mut self, low: i64, high: i64) -> i64 {
        assert!(low <= high, "integers: low > high");
        let rng = high.wrapping_sub(low) as u64;
        low.wrapping_add(self.bounded_u64(rng) as i64)
    }

    /// `rng.integers(low, high, size=n)`
    pub fn integers_n(&mut self, low: i64, high: i64, n: usize) -> Vec<i64> {
        if n == 0 {
            return Vec::new();
        }
        (0..n).map(|_| self.integers(low, high)).collect()
    }

    // ---- uniform ---------------------------------------------------------------------------------------------

    /// `rng.random()`
    pub fn random(&mut self) -> f64 {
        self.bits.next_double()
    }

    /// `rng.random(n)`
    pub fn random_n(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.bits.next_double()).collect()
    }

    // ---- normal family ---------------------------------------------------------------------------------------

    /// numpy's ziggurat `random_standard_normal`.
    pub fn standard_normal(&mut self) -> f64 {
        loop {
            let mut r = self.bits.next_u64();
            let idx = (r & 0xff) as usize;
            r >>= 8;
            let sign = r & 0x1;
            let rabs = (r >> 1) & 0x000f_ffff_ffff_ffff;
            let mut x = rabs as f64 * WI_DOUBLE[idx];
            if sign & 0x1 != 0 {
                x = -x;
            }
            if rabs < KI_DOUBLE[idx] {
                return x;
            }
            if idx == 0 {
                loop {
                    let xx = -ZIGGURAT_NOR_INV_R * (-self.bits.next_double()).ln_1p();
                    let yy = -(-self.bits.next_double()).ln_1p();
                    if yy + yy > xx * xx {
                        return if (rabs >> 8) & 0x1 != 0 { -(ZIGGURAT_NOR_R + xx) } else { ZIGGURAT_NOR_R + xx };
                    }
                }
            } else if (FI_DOUBLE[idx - 1] - FI_DOUBLE[idx]).mul_add(self.bits.next_double(), FI_DOUBLE[idx])
                < (-0.5 * x * x).exp()
            {
                return x;
            }
        }
    }

    /// `rng.normal(loc, scale)`: `loc + scale * z`, fused.
    pub fn normal(&mut self, loc: f64, scale: f64) -> f64 {
        scale.mul_add(self.standard_normal(), loc)
    }

    /// `rng.normal(loc, scale, n)`
    pub fn normal_n(&mut self, loc: f64, scale: f64, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.normal(loc, scale)).collect()
    }

    /// `rng.lognormal(mean, sigma, n)`
    pub fn lognormal_n(&mut self, mean: f64, sigma: f64, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.normal(mean, sigma).exp()).collect()
    }

    // ---- binomial --------------------------------------------------------------------------------------------

    /// `rng.binomial(n, p)`: inversion below n*p = 30, BTPE above, as numpy splits it.
    pub fn binomial(&mut self, n: i64, p: f64) -> i64 {
        assert!((0.0..=1.0).contains(&p) && n >= 0, "binomial: bad parameters");
        if n == 0 || p == 0.0 {
            return 0;
        }
        if p <= 0.5 {
            if p * n as f64 <= 30.0 { self.binomial_inversion(n, p) } else { self.binomial_btpe(n, p) }
        } else {
            let q = 1.0 - p;
            if q * n as f64 <= 30.0 { n - self.binomial_inversion(n, q) } else { n - self.binomial_btpe(n, q) }
        }
    }

    fn binomial_inversion(&mut self, n: i64, p: f64) -> i64 {
        let q = 1.0 - p;
        let qn = (n as f64 * (-p).ln_1p()).exp();
        let np = n as f64 * p;
        let limit = 10.0f64.mul_add(np.mul_add(q, 1.0).sqrt(), np);
        let bound = (if (n as f64) < limit { n as f64 } else { limit }) as i64;
        let mut x: i64 = 0;
        let mut px = qn;
        let mut u = self.bits.next_double();
        while u > px {
            x += 1;
            if x > bound {
                x = 0;
                px = qn;
                u = self.bits.next_double();
            } else {
                u -= px;
                px = ((n - x + 1) as f64 * p * px) / (x as f64 * q);
            }
        }
        x
    }

    #[allow(clippy::many_single_char_names)]
    fn binomial_btpe(&mut self, n: i64, p: f64) -> i64 {
        let r = if p < 1.0 - p { p } else { 1.0 - p };
        let q = 1.0 - r;
        let fm = (n as f64).mul_add(r, r);
        let m = fm.floor() as i64;
        let p1 = 2.195f64.mul_add((n as f64 * r * q).sqrt(), -(4.6 * q)).floor() + 0.5;
        let xm = m as f64 + 0.5;
        let xl = xm - p1;
        let xr = xm + p1;
        let c = 0.134 + 20.5 / (15.3 + m as f64);
        let mut a = (fm - xl) / (-r).mul_add(xl, fm);
        let laml = a * (1.0 + a / 2.0);
        a = (xr - fm) / (xr * q);
        let lamr = a * (1.0 + a / 2.0);
        let p2 = p1 * 2.0f64.mul_add(c, 1.0);
        let p3 = p2 + c / laml;
        let p4 = p3 + c / lamr;
        let nf = n as f64;
        let mf = m as f64;
        loop {
            // Step10
            let nrq = nf * r * q;
            let u = self.bits.next_double() * p4;
            let mut v = self.bits.next_double();
            let y: i64;
            if u <= p1 {
                y = ((-v).mul_add(p1, xm) + u).floor() as i64;
                return if p > 0.5 { n - y } else { y };
            }
            if u <= p2 {
                // Step20
                let x = xl + (u - p1) / c;
                v = v.mul_add(c, 1.0) - (mf - x + 0.5).abs() / p1;
                if v > 1.0 {
                    continue;
                }
                y = x.floor() as i64;
            } else if u <= p3 {
                // Step30
                let yf = (xl + v.ln() / laml).floor();
                y = yf as i64;
                if y < 0 || v == 0.0 {
                    continue;
                }
                v = v * (u - p2) * laml;
            } else {
                // Step40
                let yf = (xr - v.ln() / lamr).floor();
                y = yf as i64;
                if y > n || v == 0.0 {
                    continue;
                }
                v = v * (u - p3) * lamr;
            }
            // Step50
            let k = (y - m).abs();
            if !(k > 20 && (k as f64) < nrq / 2.0 - 1.0) {
                let s = r / q;
                let a = s * (nf + 1.0);
                let mut f = 1.0;
                if m < y {
                    let mut i = m + 1;
                    while i <= y {
                        f *= a / i as f64 - s;
                        i += 1;
                    }
                } else if m > y {
                    let mut i = y + 1;
                    while i <= m {
                        f /= a / i as f64 - s;
                        i += 1;
                    }
                }
                if v > f {
                    continue;
                }
                return if p > 0.5 { n - y } else { y };
            }
            // Step52
            let kf = k as f64;
            let rho = (kf / nrq) * (kf.mul_add(kf / 3.0 + 0.625, 0.16666666666666666) / nrq + 0.5);
            let t = -kf * kf / (2.0 * nrq);
            let big_a = v.ln();
            if big_a < t - rho {
                return if p > 0.5 { n - y } else { y };
            }
            if big_a > t + rho {
                continue;
            }
            let x1 = y as f64 + 1.0;
            let f1 = mf + 1.0;
            let z = nf + 1.0 - mf;
            let w = nf - y as f64 + 1.0;
            let x2 = x1 * x1;
            let f2 = f1 * f1;
            let z2 = z * z;
            let w2 = w * w;
            let head = xm.mul_add((f1 / x1).ln(), ((n - m) as f64 + 0.5) * (z / w).ln());
            let head = ((y - m) as f64).mul_add((w * r / (x1 * q)).ln(), head);
            let bound = head
                + (13680. - (462. - (132. - (99. - 140. / f2) / f2) / f2) / f2) / f1 / 166320.
                + (13680. - (462. - (132. - (99. - 140. / z2) / z2) / z2) / z2) / z / 166320.
                + (13680. - (462. - (132. - (99. - 140. / x2) / x2) / x2) / x2) / x1 / 166320.
                + (13680. - (462. - (132. - (99. - 140. / w2) / w2) / w2) / w2) / w / 166320.;
            if big_a > bound {
                continue;
            }
            return if p > 0.5 { n - y } else { y };
        }
    }

    // ---- permutations ----------------------------------------------------------------------------------------

    /// numpy's `random_interval(max)`: masked rejection in `[0, max]`, 32-bit draws when `max` fits.
    pub fn random_interval(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        let mut mask = max;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        mask |= mask >> 32;
        if max <= 0xFFFF_FFFF {
            loop {
                let value = self.bits.next_u32() as u64 & mask;
                if value <= max {
                    return value;
                }
            }
        } else {
            loop {
                let value = self.bits.next_u64() & mask;
                if value <= max {
                    return value;
                }
            }
        }
    }

    /// `rng.shuffle(x)` for a list or a 1-d array: Fisher-Yates from the end with `random_interval`.
    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        let n = x.len();
        if n < 2 {
            return;
        }
        for i in (1..n).rev() {
            let j = self.random_interval(i as u64) as usize;
            x.swap(i, j);
        }
    }

    /// `rng.permutation(n)`
    pub fn permutation(&mut self, n: usize) -> Vec<i64> {
        let mut v: Vec<i64> = (0..n as i64).collect();
        self.shuffle(&mut v);
        v
    }

    /// `_shuffle_int(n, first, data)`: the Lemire-bounded shuffle `choice` uses internally.
    fn shuffle_int(&mut self, data: &mut [i64], first: usize) {
        let n = data.len();
        for i in (first..n).rev() {
            let j = self.bounded_u64(i as u64) as usize;
            data.swap(i, j);
        }
    }

    /// `rng.choice(pop_size, size=size, replace=False)` as indices (`shuffle=True`).
    pub fn choice_without_replace(&mut self, pop_size: usize, size: usize) -> Vec<i64> {
        assert!(size <= pop_size, "choice: cannot take a larger sample than population when replace is False");
        if size == 0 {
            return Vec::new();
        }
        if pop_size > 10000 && size > pop_size / 50 {
            // tail shuffle
            let mut idx: Vec<i64> = (0..pop_size as i64).collect();
            let first = (pop_size - size).max(1);
            self.shuffle_int(&mut idx, first);
            return idx[pop_size - size..].to_vec();
        }
        // Floyd's algorithm with numpy's open-addressing set, then a full shuffle
        let set_size = (1.2 * size as f64) as u64;
        let mut mask = set_size;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        mask |= mask >> 32;
        let mut hash_set = vec![u64::MAX; (mask + 1) as usize];
        let mut out = vec![0i64; size];
        for j in (pop_size - size)..pop_size {
            let val = self.bounded_u64(j as u64);
            let mut loc = val & mask;
            while hash_set[loc as usize] != u64::MAX && hash_set[loc as usize] != val {
                loc = (loc + 1) & mask;
            }
            if hash_set[loc as usize] == u64::MAX {
                hash_set[loc as usize] = val;
                out[j + size - pop_size] = val as i64;
            } else {
                let mut loc = j as u64 & mask;
                while hash_set[loc as usize] != u64::MAX {
                    loc = (loc + 1) & mask;
                }
                hash_set[loc as usize] = j as u64;
                out[j + size - pop_size] = j as i64;
            }
        }
        self.shuffle_int(&mut out, 1);
        out
    }

    /// `rng.choice(pop_size)` (a single index, with replacement).
    pub fn choice_index(&mut self, pop_size: usize) -> usize {
        self.integers(0, pop_size as i64) as usize
    }

    /// `rng.choice(pop_size, size=n, p=p)` with replacement: cumulative sums, a uniform draw per sample and a
    /// right-sided search, as numpy does it.
    pub fn choice_p(&mut self, p: &[f64], n: usize) -> Vec<usize> {
        let mut cdf: Vec<f64> = Vec::with_capacity(p.len());
        let mut acc = 0.0;
        for &x in p {
            acc += x;
            cdf.push(acc);
        }
        let last = *cdf.last().expect("choice: empty p");
        for c in cdf.iter_mut() {
            *c /= last;
        }
        (0..n)
            .map(|_| {
                let u = self.bits.next_double();
                cdf.partition_point(|&c| c <= u)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_zero_first_doubles() {
        // np.random.default_rng(0).random(3)
        let mut g = Generator::new(0);
        let got = g.random_n(3);
        assert_eq!(got, vec![0.6369616873214543, 0.2697867137638703, 0.04097352393619469]);
    }
}
