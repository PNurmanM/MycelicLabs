//! The numpy reductions whose floating-point results the published numbers depend on.
//!
//! `np.sum`/`np.mean` add with numpy's pairwise scheme (8 accumulators, blocks of 128), not left to right, so a
//! mean of the same values can differ in the last bit from a naive loop; `np.std`, `np.percentile` and
//! `np.median` follow numpy's own operation order (`_var`, `_lerp`, `_get_indexes`).  `tests/numpy_math.rs`
//! checks each against values recorded from numpy.

/// numpy's `@TYPE@_pairwise_sum` for a contiguous float64 run.
pub fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        // numpy starts from -0.0 so a sum of negative zeros stays negative
        let mut res = -0.0;
        for &x in a {
            res += x;
        }
        res
    } else if n <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

/// `np.add.reduce` over a float64 vector: the reduction starts from the identity 0.0.
pub fn sum(a: &[f64]) -> f64 {
    0.0 + pairwise_sum(a)
}

/// `np.add.reduce(x, dtype=float64)` over an integer vector: the cast goes through numpy's 8192-element buffer,
/// and each buffer is reduced pairwise into the running total.
pub fn sum_int_as_f64(a: &[i64]) -> f64 {
    let mut total = 0.0;
    for chunk in a.chunks(8192) {
        let buf: Vec<f64> = chunk.iter().map(|&v| v as f64).collect();
        total += pairwise_sum(&buf);
    }
    total
}

/// `np.mean` of float64 values (NaN for an empty input, as numpy returns with a warning).
pub fn mean(a: &[f64]) -> f64 {
    sum(a) / a.len() as f64
}

/// `np.mean` of integer values.
pub fn mean_int(a: &[i64]) -> f64 {
    sum_int_as_f64(a) / a.len() as f64
}

/// `np.std(a, ddof=ddof)` of float64 values.
pub fn std(a: &[f64], ddof: usize) -> f64 {
    let n = a.len();
    let m = sum(a) / n as f64;
    let sq: Vec<f64> = a
        .iter()
        .map(|&x| {
            let d = x - m;
            d * d
        })
        .collect();
    let div = n.saturating_sub(ddof) as f64;
    (sum(&sq) / div).sqrt()
}

/// `np.std` of integer values (numpy computes in float64).
pub fn std_int(a: &[i64], ddof: usize) -> f64 {
    let f: Vec<f64> = a.iter().map(|&v| v as f64).collect();
    // the mean is taken through the buffered integer cast, the deviations in float64
    let n = a.len();
    let m = sum_int_as_f64(a) / n as f64;
    let sq: Vec<f64> = f
        .iter()
        .map(|&x| {
            let d = x - m;
            d * d
        })
        .collect();
    let div = n.saturating_sub(ddof) as f64;
    (sum(&sq) / div).sqrt()
}

/// numpy's `_lerp(a, b, t)`: `a + (b-a)*t`, or `b - (b-a)*(1-t)` when `t >= 0.5`.
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    let diff = b - a;
    if t >= 0.5 { b - diff * (1.0 - t) } else { a + diff * t }
}

/// `np.percentile(a, q)` with the default linear method, `q` a Python scalar in percent.
pub fn percentile(a: &[f64], q: f64) -> f64 {
    let mut v: Vec<f64> = a.to_vec();
    v.sort_by(|x, y| x.partial_cmp(y).expect("percentile: NaN"));
    quantile_sorted(&v, q / 100.0)
}

/// `np.percentile` of integer values.
pub fn percentile_int(a: &[i64], q: f64) -> f64 {
    let v: Vec<f64> = a.iter().map(|&x| x as f64).collect();
    percentile(&v, q)
}

/// Linear-method quantile of sorted values (`_quantile` with `_get_indexes`/`_get_gamma`).
pub fn quantile_sorted(v: &[f64], quantile: f64) -> f64 {
    let n = v.len();
    assert!(n > 0, "quantile of an empty array");
    let virtual_index = (n as f64 - 1.0) * quantile;
    let (prev, next) = if virtual_index >= n as f64 - 1.0 {
        (n - 1, n - 1)
    } else if virtual_index < 0.0 {
        (0, 0)
    } else {
        let p = virtual_index.floor() as usize;
        (p, p + 1)
    };
    // gamma is measured from the (possibly clamped to -1) previous index, exactly as numpy does
    let prev_for_gamma = if virtual_index >= n as f64 - 1.0 {
        -1.0
    } else if virtual_index < 0.0 {
        0.0
    } else {
        virtual_index.floor()
    };
    let gamma = virtual_index - prev_for_gamma;
    lerp(v[prev], v[next], gamma)
}

/// `np.median` of float64 values: the middle value, or the mean of the middle two.
pub fn median(a: &[f64]) -> f64 {
    let mut v: Vec<f64> = a.to_vec();
    v.sort_by(|x, y| x.partial_cmp(y).expect("median: NaN"));
    let n = v.len();
    assert!(n > 0, "median of an empty array");
    let szh = n / 2;
    if n % 2 == 1 { mean(&v[szh..szh + 1]) } else { mean(&v[szh - 1..szh + 1]) }
}

/// `np.median` of integer values (a float).
pub fn median_int(a: &[i64]) -> f64 {
    let v: Vec<f64> = a.iter().map(|&x| x as f64).collect();
    median(&v)
}

/// `np.linspace(start, stop, num)`
pub fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 0 {
        return Vec::new();
    }
    if num == 1 {
        return vec![start];
    }
    let div = (num - 1) as f64;
    let delta = stop - start;
    let step = delta / div;
    let mut out: Vec<f64> = (0..num)
        .map(|i| if step == 0.0 { (i as f64 / div) * delta + start } else { i as f64 * step + start })
        .collect();
    out[num - 1] = stop;
    out
}

/// `np.argmax` over a slice: the first maximum.
pub fn argmax(a: &[f64]) -> usize {
    let mut best = 0;
    for (i, &x) in a.iter().enumerate() {
        if x > a[best] {
            best = i;
        }
    }
    best
}

/// `np.argsort(a, kind="stable")`
pub fn argsort_stable<T: PartialOrd>(a: &[T]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..a.len()).collect();
    idx.sort_by(|&i, &j| a[i].partial_cmp(&a[j]).expect("argsort: unordered values"));
    idx
}

/// `np.argsort(-x)` for values without ties: a descending order (ties, which the callers rule out by
/// construction, fall back to index order).
pub fn argsort_desc(a: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..a.len()).collect();
    idx.sort_by(|&i, &j| (-a[i]).partial_cmp(&-a[j]).expect("argsort: NaN"));
    idx
}

/// `np.unique(a, return_index=True, return_counts=True)` for sorted-able integers: (values, first index,
/// counts), values ascending.
pub fn unique_index_counts(a: &[i64]) -> (Vec<i64>, Vec<usize>, Vec<usize>) {
    let order = argsort_stable(a);
    let mut vals = Vec::new();
    let mut first = Vec::new();
    let mut counts: Vec<usize> = Vec::new();
    for &i in &order {
        let v = a[i];
        if vals.last() == Some(&v) {
            *counts.last_mut().unwrap() += 1;
        } else {
            vals.push(v);
            first.push(i);
            counts.push(1);
        }
    }
    (vals, first, counts)
}

/// `np.unique(a, return_inverse=True, return_counts=True)`: (values, inverse, counts).
pub fn unique_inverse_counts(a: &[i64]) -> (Vec<i64>, Vec<usize>, Vec<usize>) {
    let order = argsort_stable(a);
    let mut vals: Vec<i64> = Vec::new();
    let mut counts: Vec<usize> = Vec::new();
    let mut inv = vec![0usize; a.len()];
    for &i in &order {
        let v = a[i];
        if vals.last() != Some(&v) {
            vals.push(v);
            counts.push(0);
        }
        *counts.last_mut().unwrap() += 1;
        inv[i] = vals.len() - 1;
    }
    (vals, inv, counts)
}

/// `np.bincount(a, minlength)`
pub fn bincount(a: &[i64], minlength: usize) -> Vec<i64> {
    let len = a.iter().map(|&v| v as usize + 1).max().unwrap_or(0).max(minlength);
    let mut out = vec![0i64; len];
    for &v in a {
        out[v as usize] += 1;
    }
    out
}

/// `np.rint` (round half to even)
pub fn rint(x: f64) -> f64 {
    x.round_ties_even()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_sums_keep_order() {
        assert_eq!(pairwise_sum(&[0.1, 0.2, 0.3]), 0.1 + 0.2 + 0.3);
        assert!(pairwise_sum(&[]).is_sign_negative());
        assert_eq!(sum(&[]), 0.0);
    }

    #[test]
    fn percentile_endpoints() {
        assert_eq!(percentile(&[1.0, 2.0, 3.0, 4.0], 50.0), 2.5);
        assert_eq!(percentile(&[5.0], 90.0), 5.0);
        assert_eq!(percentile_int(&[1, 2, 3, 4, 5], 100.0), 5.0);
    }
}
