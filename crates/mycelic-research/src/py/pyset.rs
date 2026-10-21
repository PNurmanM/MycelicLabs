//! A Python `set` of ints that iterates in CPython 3.11's order.
//!
//! Python's set iteration order is the order of its open-addressing hash table, which depends on the values,
//! the insertion history and every resize.  The simulator iterates sets in a few places where the order is
//! observable - the region sets a synthesis tallies (`max(rc, key=rc.get)` breaks ties by insertion order) and the
//! triage seeds a descent starts from (their order decides which site is routed first, and so the random draws
//! of everything after it) - so those sets are this type: a transcription of `Objects/setobject.c` (linear probing
//! over 9 slots, perturbed probing, `fill*5 >= mask*3` growth, the merge fast paths).  Elements are never removed
//! in the simulator, so dummy entries do not arise.

const MINSIZE: usize = 8;
const LINEAR_PROBES: usize = 9;
const PERTURB_SHIFT: u32 = 5;

/// `hash(int)` for the magnitudes the simulator uses (|x| < 2**61 - 1): the value itself, except -1 -> -2.
#[inline]
fn py_hash(x: i64) -> u64 {
    if x == -1 { (-2i64) as u64 } else { x as u64 }
}

#[derive(Clone, Debug)]
pub struct PySet {
    table: Vec<Option<i64>>,
    mask: usize,
    used: usize,
}

impl Default for PySet {
    fn default() -> Self {
        Self::new()
    }
}

impl PySet {
    /// `set()`
    pub fn new() -> Self {
        PySet { table: vec![None; MINSIZE], mask: MINSIZE - 1, used: 0 }
    }

    /// `{x}`
    pub fn from_one(x: i64) -> Self {
        let mut s = PySet::new();
        s.add(x);
        s
    }

    /// `set(iterable)` over a list or any non-set iterable: element by element.
    pub fn from_iter_seq<I: IntoIterator<Item = i64>>(it: I) -> Self {
        let mut s = PySet::new();
        for x in it {
            s.add(x);
        }
        s
    }

    /// `set(d)` over a dict's keys: one resize up front for the dict's size, then insertions.
    pub fn from_dict_keys(keys: &[i64]) -> Self {
        let mut s = PySet::new();
        let dictsize = keys.len();
        if (s.used + dictsize) * 5 >= s.mask * 3 {
            s.resize((s.used + dictsize) * 2);
        }
        for &k in keys {
            s.add(k);
        }
        s
    }

    pub fn len(&self) -> usize {
        self.used
    }

    pub fn is_empty(&self) -> bool {
        self.used == 0
    }

    pub fn contains(&self, x: i64) -> bool {
        let hash = py_hash(x);
        let mask = self.mask;
        let mut i = hash as usize & mask;
        let mut perturb = hash;
        loop {
            let probes = if i + LINEAR_PROBES <= mask { LINEAR_PROBES } else { 0 };
            for j in 0..=probes {
                match self.table[i + j] {
                    None => return false,
                    Some(k) if k == x => return true,
                    _ => {}
                }
            }
            perturb >>= PERTURB_SHIFT;
            i = (i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb as usize)) & mask;
        }
    }

    /// `s.add(x)` (`set_add_entry`)
    pub fn add(&mut self, x: i64) {
        let hash = py_hash(x);
        let mask = self.mask;
        let mut i = hash as usize & mask;
        let mut perturb = hash;
        loop {
            let probes = if i + LINEAR_PROBES <= mask { LINEAR_PROBES } else { 0 };
            for j in 0..=probes {
                match self.table[i + j] {
                    None => {
                        self.table[i + j] = Some(x);
                        self.used += 1;
                        // no dummies, so fill == used
                        if self.used * 5 >= mask * 3 {
                            let target = if self.used > 50000 { self.used * 2 } else { self.used * 4 };
                            self.resize(target);
                        }
                        return;
                    }
                    Some(k) if k == x => return,
                    _ => {}
                }
            }
            perturb >>= PERTURB_SHIFT;
            i = (i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb as usize)) & mask;
        }
    }

    fn insert_clean(table: &mut [Option<i64>], mask: usize, x: i64) {
        let hash = py_hash(x);
        let mut i = hash as usize & mask;
        let mut perturb = hash;
        loop {
            if table[i].is_none() {
                table[i] = Some(x);
                return;
            }
            if i + LINEAR_PROBES <= mask {
                for j in 1..=LINEAR_PROBES {
                    if table[i + j].is_none() {
                        table[i + j] = Some(x);
                        return;
                    }
                }
            }
            perturb >>= PERTURB_SHIFT;
            i = (i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb as usize)) & mask;
        }
    }

    /// `set_table_resize(so, minused)`
    fn resize(&mut self, minused: usize) {
        let mut newsize = MINSIZE;
        while newsize <= minused {
            newsize <<= 1;
        }
        let old = std::mem::replace(&mut self.table, vec![None; newsize]);
        self.mask = newsize - 1;
        for x in old.into_iter().flatten() {
            Self::insert_clean(&mut self.table, self.mask, x);
        }
    }

    /// `s |= other` / `s.update(other)` with `other` a set (`set_merge`).
    pub fn update_set(&mut self, other: &PySet) {
        if other.used == 0 {
            return;
        }
        if (self.used + other.used) * 5 >= self.mask * 3 {
            self.resize((self.used + other.used) * 2);
        }
        if self.used == 0 && self.mask == other.mask {
            // empty target of the same size: the table is copied slot for slot
            self.table.clone_from(&other.table);
            self.used = other.used;
            return;
        }
        if self.used == 0 {
            let mask = self.mask;
            for x in other.table.iter().flatten() {
                Self::insert_clean(&mut self.table, mask, *x);
            }
            self.used = other.used;
            return;
        }
        for x in other.table.iter().flatten() {
            self.add(*x);
        }
    }

    /// `set(other_set)` / `other.copy()`
    pub fn copy_of(other: &PySet) -> Self {
        let mut s = PySet::new();
        s.update_set(other);
        s
    }

    /// `a | b`
    pub fn union(&self, other: &PySet) -> PySet {
        let mut s = PySet::copy_of(self);
        s.update_set(other);
        s
    }

    /// Iteration in table order, which is Python's iteration order.
    pub fn iter(&self) -> impl Iterator<Item = i64> + '_ {
        self.table.iter().flatten().copied()
    }

    pub fn to_vec(&self) -> Vec<i64> {
        self.iter().collect()
    }

    /// `len(a & b)`
    pub fn intersection_len(&self, other: &PySet) -> usize {
        let (small, big) = if self.used <= other.used { (self, other) } else { (other, self) };
        small.iter().filter(|&x| big.contains(x)).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_ints_iterate_in_slot_order() {
        let s = PySet::from_iter_seq([5, 3, 1]);
        assert_eq!(s.to_vec(), vec![1, 3, 5]);
        let s = PySet::from_iter_seq([9, 1]);
        // 9 & 7 == 1 collides with 1 inserted later: 9 sits in slot 1, 1 probes linearly... both orders matter
        assert_eq!(s.len(), 2);
    }
}
