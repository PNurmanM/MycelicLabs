//! Heterogeneous synthetic enterprise org-chart generator (`org.py`).
//!
//! Six levels, USER(0) -> TEAM(1) -> DEPARTMENT(2) -> SITE/SUBSIDIARY(3) -> REGION(4) -> ENTERPRISE(5).  Fan-in is
//! drawn from heavy-tailed distributions clipped to plausible ranges, so the tree is genuinely heterogeneous
//! (small teams next to large ones, satellite offices next to major sites), not a balanced k-ary tree.
//! Deterministic given (seed, target users, fan-in spec).

use crate::np::Generator;
use crate::np::math;
use crate::py::json::round;
use serde_json::{Map, Value, json};

pub const LEVEL_NAMES: [&str; 6] = ["user", "team", "department", "site", "region", "enterprise"];
pub const USER: usize = 0;
pub const TEAM: usize = 1;
pub const DEPT: usize = 2;
pub const SITE: usize = 3;
pub const REGION: usize = 4;
pub const ENT: usize = 5;

pub const N_FUNCTIONS: i64 = 8;
pub const FUNCTION_NAMES: [&str; 8] =
    ["engineering", "operations", "supply-chain", "sales", "finance", "security", "support", "manufacturing"];

#[derive(Clone, Debug)]
pub struct Node {
    pub nid: usize,
    pub level: usize,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub name: String,
    /// leaf users beneath this node (1 for a user)
    pub n_users: usize,
    /// functional family (eng, ops, sales, finance, ...)
    pub function: i64,
}

/// One fan-in law: (median, sigma, lo, hi) of a clipped lognormal.
pub type Fanin = (f64, f64, i64, i64);

/// The default heterogeneous fan-in spec, indexed by the child level (TEAM = users per team, ...).
pub fn default_fanin() -> [Fanin; 6] {
    let mut f = [(0.0, 0.0, 0, 0); 6];
    f[TEAM] = (9.0, 0.32, 4, 22); // users per team
    f[DEPT] = (7.0, 0.45, 3, 18); // teams per department
    f[SITE] = (5.0, 0.60, 1, 20); // departments per site
    f[REGION] = (6.0, 0.70, 1, 26); // sites per region
    f[ENT] = (7.0, 0.0, 5, 9); // regions per enterprise
    f
}

#[derive(Clone, Debug)]
pub struct Org {
    pub nodes: Vec<Node>,
    /// node ids per level
    pub levels: [Vec<usize>; 6],
    pub root: usize,
    pub seed: u64,
    pub user_ids: Vec<usize>,
    /// ancestor at each level, precomputed (`ancestor_at` is called per extracted claim)
    anc: Vec<[Option<usize>; 6]>,
}

impl Org {
    pub fn parent_of(&self, nid: usize) -> Option<usize> {
        self.nodes[nid].parent
    }

    pub fn ancestors(&self, nid: usize) -> Vec<usize> {
        let mut out = Vec::new();
        let mut p = self.nodes[nid].parent;
        while let Some(x) = p {
            out.push(x);
            p = self.nodes[x].parent;
        }
        out
    }

    /// The node itself at its own level, else its ancestor at `level`.
    pub fn ancestor_at(&self, nid: usize, level: usize) -> Option<usize> {
        self.anc[nid][level]
    }

    pub fn leaves_under(&self, nid: usize) -> Vec<usize> {
        if self.nodes[nid].level == USER {
            return vec![nid];
        }
        let mut out = Vec::new();
        let mut stack = vec![nid];
        while let Some(c) = stack.pop() {
            if self.nodes[c].level == USER {
                out.push(c);
            } else {
                stack.extend(self.nodes[c].children.iter().copied());
            }
        }
        out
    }

    /// `Org.stats()`
    pub fn stats(&self) -> Value {
        let mut s = Map::new();
        s.insert("n_users".into(), json!(self.user_ids.len()));
        s.insert("seed".into(), json!(self.seed));
        for (l, level) in self.levels.iter().enumerate().skip(1) {
            let fan: Vec<i64> = level.iter().map(|&i| self.nodes[i].children.len() as i64).collect();
            if fan.is_empty() {
                continue;
            }
            s.insert(
                LEVEL_NAMES[l].into(),
                json!({
                    "count": level.len(),
                    "fanin_mean": round(math::mean_int(&fan), 2),
                    "fanin_p10": math::percentile_int(&fan, 10.0) as i64,
                    "fanin_p50": math::percentile_int(&fan, 50.0) as i64,
                    "fanin_p90": math::percentile_int(&fan, 90.0) as i64,
                    "fanin_min": *fan.iter().min().unwrap(),
                    "fanin_max": *fan.iter().max().unwrap(),
                }),
            );
        }
        let usz: Vec<i64> = self.levels[SITE].iter().map(|&i| self.nodes[i].n_users as i64).collect();
        s.insert(
            "site_user_counts".into(),
            json!({
                "min": *usz.iter().min().unwrap(),
                "p50": math::percentile_int(&usz, 50.0) as i64,
                "max": *usz.iter().max().unwrap(),
            }),
        );
        let mut rsz: Vec<i64> = self.levels[REGION].iter().map(|&i| self.nodes[i].n_users as i64).collect();
        rsz.sort();
        s.insert("region_user_counts".into(), json!(rsz));
        Value::Object(s)
    }
}

/// n draws with median ~mean, clipped to [lo, hi], integer.
fn clipped_lognormal(rng: &mut Generator, n: usize, spec: Fanin) -> Vec<i64> {
    let (mean, sigma, lo, hi) = spec;
    let mu = mean.ln();
    rng.lognormal_n(mu, sigma, n).into_iter().map(|v| (math::rint(v).clamp(lo as f64, hi as f64)) as i64).collect()
}

struct Builder {
    nodes: Vec<Node>,
    levels: [Vec<usize>; 6],
}

impl Builder {
    fn add(&mut self, level: usize, parent: Option<usize>, name: String, function: i64) -> usize {
        let nid = self.nodes.len();
        self.nodes.push(Node { nid, level, parent, children: Vec::new(), name, n_users: 0, function });
        self.levels[level].push(nid);
        if let Some(p) = parent {
            self.nodes[p].children.push(nid);
        }
        nid
    }
}

fn remove_value(v: &mut Vec<usize>, x: usize) {
    if let Some(pos) = v.iter().position(|&y| y == x) {
        v.remove(pos);
    }
}

/// `build_org(target_users, seed, fanin, n_regions)`: grow sites until each region's user target is met.
pub fn build_org(target_users: usize, seed: u64, fanin: Option<[Fanin; 6]>, n_regions: usize) -> Org {
    let mut rng = Generator::new(seed);
    let fanin = fanin.unwrap_or_else(default_fanin);
    let mut b = Builder { nodes: Vec::new(), levels: Default::default() };
    let root = b.add(ENT, None, "ENTERPRISE".into(), 0);

    // Region weights: deliberately skewed (Zipf-ish), so one region holds ~30% of the company and the smallest
    // holds ~3%.
    let w: Vec<f64> = (0..n_regions).map(|i| 1.0 / crate::py::pow((i + 1) as f64, 0.85)).collect();
    let wsum = math::sum(&w);
    let region_targets: Vec<i64> =
        w.iter().map(|&x| (math::rint(x / wsum * target_users as f64)).max(40.0) as i64).collect();

    for (ri, &region_target) in region_targets.iter().enumerate() {
        let rid = b.add(REGION, Some(root), format!("REGION-{ri:02}"), 0);
        let mut remaining = region_target;
        let mut si = 0usize;
        while remaining > 0 {
            // site size heterogeneity: mixture of satellite offices and hubs
            let u = rng.random();
            let mut site_target = if u < 0.45 {
                rng.integers(40, 220) // satellite
            } else if u < 0.85 {
                rng.integers(220, 900) // regular site
            } else {
                rng.integers(900, 4200) // major hub
            };
            site_target = if remaining > 60 { site_target.min(remaining) } else { remaining };
            let sid = b.add(SITE, Some(rid), format!("R{ri:02}-SITE-{si:02}"), 0);
            si += 1;
            let mut srem = site_target;
            let mut di = 0usize;
            while srem > 0 {
                // department = a functional family at this site
                let fnc = rng.integers(0, N_FUNCTIONS);
                let did = b.add(DEPT, Some(sid), format!("R{ri:02}S{:02}-DEPT-{di:02}", si - 1), fnc);
                di += 1;
                let n_teams = clipped_lognormal(&mut rng, 1, fanin[DEPT])[0];
                let drem_target = srem;
                let mut built = 0i64;
                for t in 0..n_teams {
                    if built >= drem_target {
                        break;
                    }
                    let mut tsz = clipped_lognormal(&mut rng, 1, fanin[TEAM])[0];
                    tsz = tsz.min((drem_target - built).max(1));
                    let tname = format!("{}-T{t:02}", b.nodes[did].name);
                    let tid = b.add(TEAM, Some(did), tname, fnc);
                    for k in 0..tsz {
                        let uname = format!("{}-U{k:02}", b.nodes[tid].name);
                        b.add(USER, Some(tid), uname, fnc);
                    }
                    built += tsz;
                }
                if built == 0 {
                    // degenerate department, drop it
                    remove_value(&mut b.nodes[sid].children, did);
                    remove_value(&mut b.levels[DEPT], did);
                }
                srem -= built;
                if built == 0 {
                    break;
                }
            }
            remaining -= site_target - srem.max(0);
            if site_target <= 0 {
                break;
            }
        }
    }

    let Builder { mut nodes, levels } = b;
    // propagate function labels upward (site/region = modal child function)
    for l in [SITE, REGION] {
        for &nid in &levels[l] {
            let fs: Vec<i64> = nodes[nid].children.iter().map(|&c| nodes[c].function).collect();
            nodes[nid].function = if fs.is_empty() {
                0
            } else {
                let counts = math::bincount(&fs, 0);
                let mut best = 0usize;
                for (i, &c) in counts.iter().enumerate() {
                    if c > counts[best] {
                        best = i;
                    }
                }
                best as i64
            };
        }
    }

    let user_ids = levels[USER].clone();
    for &nid in &user_ids {
        nodes[nid].n_users = 1;
    }
    for level in &levels[1..6] {
        for &nid in level {
            let n: usize = nodes[nid].children.iter().map(|&c| nodes[c].n_users).sum();
            nodes[nid].n_users = n;
        }
    }

    let mut anc = vec![[None; 6]; nodes.len()];
    for (nid, slots) in anc.iter_mut().enumerate() {
        let mut p = Some(nid);
        while let Some(x) = p {
            let lvl = nodes[x].level;
            if slots[lvl].is_none() {
                slots[lvl] = Some(x);
            }
            p = nodes[x].parent;
        }
    }
    Org { nodes, levels, root, seed, user_ids, anc }
}

/// `build_org` with the default fan-in and seven regions.
pub fn build_org_default(target_users: usize, seed: u64) -> Org {
    build_org(target_users, seed, None, 7)
}

/// `rescale_fanin(base, level, median)`
pub fn rescale_fanin(base: [Fanin; 6], level: usize, median: f64) -> [Fanin; 6] {
    let mut out = base;
    out[level].0 = median;
    out
}
