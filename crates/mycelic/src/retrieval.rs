//! Retrieval over organizational memory: BM25 keyword ranking scoped to what the caller may see.
//!
//! The index is built per organization from every active memory and rebuilt lazily when the store's revision
//! changes.  Ranking is BM25 with a mild boost for higher organizational layers, so an enterprise conclusion
//! outranks one of the raw observations it was built from when both match the query, while a precise raw
//! observation still wins when the conclusion does not mention the query's terms.  Visibility is a predicate
//! evaluated per row (see `auth::memory_visible`), so an agent never sees a hit it could not `GET`.
//!
//! BM25 statistics (IDF, average document length) are computed over the caller's view only, so memories the caller
//! cannot read influence neither the results nor their order.  The index keeps each document's term counts;
//! a query computes document frequencies for its own terms over the view, which is the same arithmetic, in the
//! same order, as building a full BM25 model over the view (the Python implementation did that per query), and
//! therefore the same scores to the last bit.
//!
//! Embeddings are deliberately not part of this slice: the deployment must run without any model server.

use crate::hierarchy::{LAYERS, is_ancestor_or_self, layer_pos};
use crate::models::Memory;
use crate::pyjson::{fmt_fixed, round};
use crate::textutil::tokenize;
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

/// Per layer above 'agent'.
pub const LAYER_BOOST: f64 = 0.15;
pub const K1: f64 = 1.5;
pub const B: f64 = 0.75;

#[derive(Clone, Debug)]
pub struct Hit {
    pub memory: Memory,
    pub score: f64,
    pub bm25: f64,
    pub explanation: String,
}

impl Hit {
    pub fn to_json(&self) -> Value {
        json!({"memory": self.memory.to_json(), "score": round(self.score, 4), "bm25": round(self.bm25, 4),
               "explanation": self.explanation})
    }
}

fn doc_tokens(m: &Memory) -> Vec<String> {
    let parts = [
        m.text.as_str(),
        m.topic.as_deref().unwrap_or(""),
        m.entity.as_deref().unwrap_or(""),
        m.slot.as_deref().unwrap_or(""),
        m.kind.as_str(),
    ];
    let toks = tokenize(&parts.join(" "));
    if toks.is_empty() { vec!["_empty_".to_string()] } else { toks }
}

/// Okapi BM25 with the Lucene-style IDF `log(1 + (N - df + 0.5) / (df + 0.5))`, which is always positive.
///
/// `rank_bm25` was tried first in the Python implementation; its IDF goes negative for terms present in most
/// documents, which makes every score of a one- or two-memory organization negative and the retriever return
/// nothing for a brand-new deployment.
pub struct Bm25<'a> {
    pub k1: f64,
    pub b: f64,
    tf: Vec<&'a HashMap<u32, u32>>,
    len: Vec<usize>,
    avg: f64,
    idf: HashMap<u32, f64>,
}

impl<'a> Bm25<'a> {
    /// A model over `docs` (term-id counts and lengths) that can score the query terms in `query`.
    pub fn new(docs: &[(&'a HashMap<u32, u32>, usize)], query: &[u32], k1: f64, b: f64) -> Bm25<'a> {
        let n = docs.len();
        let total: usize = docs.iter().map(|d| d.1).sum();
        let avg = if n > 0 { total as f64 / n as f64 } else { 0.0 };
        let wanted: HashSet<u32> = query.iter().copied().collect();
        let mut df: HashMap<u32, usize> = wanted.iter().map(|t| (*t, 0)).collect();
        for (tf, _) in docs {
            for t in &wanted {
                if tf.contains_key(t) {
                    *df.get_mut(t).expect("query term") += 1;
                }
            }
        }
        let idf = df
            .into_iter()
            .filter(|(_, f)| *f > 0)
            .map(|(t, f)| (t, (1.0 + (n as f64 - f as f64 + 0.5) / (f as f64 + 0.5)).ln()))
            .collect();
        Bm25 { k1, b, tf: docs.iter().map(|d| d.0).collect(), len: docs.iter().map(|d| d.1).collect(), avg, idf }
    }

    pub fn get_scores(&self, query: &[u32]) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.tf.len());
        for (tf, &dl) in self.tf.iter().zip(&self.len) {
            let mut score = 0.0;
            let ratio = if self.avg != 0.0 { dl as f64 / self.avg } else { 1.0 };
            let norm = self.k1 * (1.0 - self.b + self.b * ratio);
            for t in query {
                if let Some(&f) = tf.get(t)
                    && f > 0
                {
                    let f = f as f64;
                    score += self.idf[t] * f * (self.k1 + 1.0) / (f + norm);
                }
            }
            out.push(score);
        }
        out
    }
}

/// Term-id interning shared by an index and the queries against it.
#[derive(Default)]
pub struct Vocab {
    ids: HashMap<String, u32>,
}

impl Vocab {
    fn intern(&mut self, t: String) -> u32 {
        let next = self.ids.len() as u32;
        *self.ids.entry(t).or_insert(next)
    }

    /// Query tokens as ids; a token no document has gets a fresh id that matches nothing.
    pub fn lookup(&self, toks: &[String]) -> Vec<u32> {
        toks.iter().map(|t| self.ids.get(t).copied().unwrap_or(u32::MAX)).collect()
    }
}

/// Tokens of every active memory of an organization; the BM25 statistics are computed per caller view.
pub struct OrgIndex {
    pub rows: Vec<Memory>,
    tf: Vec<HashMap<u32, u32>>,
    len: Vec<usize>,
    vocab: Vocab,
}

impl OrgIndex {
    pub fn new(rows: Vec<Memory>) -> OrgIndex {
        let mut vocab = Vocab::default();
        let mut tf = Vec::with_capacity(rows.len());
        let mut len = Vec::with_capacity(rows.len());
        for m in &rows {
            let toks = doc_tokens(m);
            len.push(toks.len());
            let mut counts: HashMap<u32, u32> = HashMap::new();
            for t in toks {
                *counts.entry(vocab.intern(t)).or_insert(0) += 1;
            }
            tf.push(counts);
        }
        OrgIndex { rows, tf, len, vocab }
    }
}

/// What a search asks for (the optional `POST /query` fields).
#[derive(Clone, Debug)]
pub struct SearchParams<'a> {
    pub scope: Option<&'a str>,
    pub min_layer: &'a str,
    pub k: usize,
    pub topic: Option<&'a str>,
    pub entity: Option<&'a str>,
}

impl Default for SearchParams<'_> {
    fn default() -> Self {
        SearchParams { scope: None, min_layer: "agent", k: 10, topic: None, entity: None }
    }
}

/// Rank the memories of `index` the caller may see.
pub fn search_index(index: &OrgIndex, query: &str, visible: &dyn Fn(&Memory) -> bool, p: &SearchParams) -> Vec<Hit> {
    if index.rows.is_empty() {
        return Vec::new();
    }
    let min_idx = layer_pos(p.min_layer);
    let q_tokens = tokenize(query);
    let q = index.vocab.lookup(&q_tokens);
    let view: Vec<usize> = (0..index.rows.len()).filter(|&i| visible(&index.rows[i])).collect();
    if view.is_empty() {
        return Vec::new();
    }
    let scores: Vec<f64> = if q.is_empty() {
        vec![0.0; view.len()]
    } else {
        let docs: Vec<(&HashMap<u32, u32>, usize)> = view.iter().map(|&i| (&index.tf[i], index.len[i])).collect();
        Bm25::new(&docs, &q, K1, B).get_scores(&q)
    };
    // rank row indices; only the k hits returned are cloned and explained
    struct Candidate {
        row: usize,
        score: f64,
        bm25: f64,
        layer: usize,
    }
    let mut candidates: Vec<Candidate> = Vec::new();
    for (&i, &s) in view.iter().zip(&scores) {
        let m = &index.rows[i];
        if let Some(scope) = p.scope.filter(|s| !s.is_empty())
            && !is_ancestor_or_self(scope, &m.scope)
        {
            continue;
        }
        let li = layer_pos(&m.layer);
        if li < min_idx {
            continue;
        }
        if let Some(t) = p.topic.filter(|t| !t.is_empty())
            && m.topic.as_deref() != Some(t)
        {
            continue;
        }
        if let Some(e) = p.entity.filter(|e| !e.is_empty())
            && m.entity.as_deref() != Some(e)
        {
            continue;
        }
        if !q.is_empty() && s <= 0.0 {
            continue;
        }
        let boost = 1.0 + LAYER_BOOST * li as f64;
        let score = if !q.is_empty() { s * boost } else { boost * (1.0 + m.confidence) };
        candidates.push(Candidate { row: i, score, bm25: s, layer: li });
    }
    // a memory promoted unchanged to a single-child parent unit is the same knowledge: keep the highest copy only
    let promoted_away: HashSet<&str> = candidates
        .iter()
        .map(|c| &index.rows[c.row])
        .filter(|m| m.layer != "agent")
        .filter_map(|m| m.metadata.get("promoted_from").and_then(Value::as_str).filter(|s| !s.is_empty()))
        .collect();
    candidates.retain(|c| !promoted_away.contains(index.rows[c.row].memory_id.as_str()));
    // (-score, -layer, created_at, memory_id): memory ids are unique, so this is a total order and selecting the k
    // smallest before sorting them gives exactly the order a full sort would
    let order = |a: &Candidate, b: &Candidate| {
        let (ma, mb) = (&index.rows[a.row], &index.rows[b.row]);
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.layer.cmp(&a.layer))
            .then_with(|| ma.created_at.cmp(&mb.created_at))
            .then_with(|| ma.memory_id.cmp(&mb.memory_id))
    };
    let k = p.k.max(1);
    if candidates.len() > k {
        candidates.select_nth_unstable_by(k - 1, order);
        candidates.truncate(k);
    }
    candidates.sort_by(order);
    candidates
        .into_iter()
        .map(|c| {
            let m = &index.rows[c.row];
            let explanation = if !q.is_empty() {
                let boost = 1.0 + LAYER_BOOST * c.layer as f64;
                format!("bm25={} × layer boost {} ({})", fmt_fixed(c.bm25, 3), fmt_fixed(boost, 2), m.layer)
            } else {
                format!("no query terms: ranked by layer and confidence ({})", m.layer)
            };
            Hit { memory: m.clone(), score: c.score, bm25: c.bm25, explanation }
        })
        .collect()
}

/// The per-organization index cache, keyed by the store revision it was built at.
#[derive(Default)]
pub struct Retriever {
    cache: Mutex<HashMap<String, (u64, Arc<OrgIndex>)>>,
}

impl Retriever {
    pub fn new() -> Retriever {
        Retriever::default()
    }

    /// The index for `org_id`, rebuilt from `load` when the revision moved.  The revision is read before the rows,
    /// so an index built from rows a newer commit already changed is cached under the older revision and rebuilt.
    pub fn index(
        &self,
        org_id: &str,
        revision: u64,
        load: impl FnOnce() -> anyhow::Result<Vec<Memory>>,
    ) -> anyhow::Result<Arc<OrgIndex>> {
        if let Some((rev, idx)) = self.cache.lock().unwrap_or_else(|e| e.into_inner()).get(org_id)
            && *rev == revision
        {
            return Ok(idx.clone());
        }
        let idx = Arc::new(OrgIndex::new(load()?));
        self.cache.lock().unwrap_or_else(|e| e.into_inner()).insert(org_id.to_string(), (revision, idx.clone()));
        Ok(idx)
    }

    pub fn invalidate(&self) {
        self.cache.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }
}

/// Layer names in order, for callers that print them.
pub fn layers() -> &'static [&'static str] {
    &LAYERS
}

/// BM25 scores of token lists, for callers that hold tokens rather than memories (tests, benchmarks).
pub fn bm25_scores(docs: &[Vec<String>], query: &[String], k1: f64, b: f64) -> Vec<f64> {
    let mut vocab = Vocab::default();
    let counts: Vec<(HashMap<u32, u32>, usize)> = docs
        .iter()
        .map(|d| {
            let mut c = HashMap::new();
            for t in d {
                *c.entry(vocab.intern(t.clone())).or_insert(0) += 1;
            }
            (c, d.len())
        })
        .collect();
    let q = vocab.lookup(query);
    let refs: Vec<(&HashMap<u32, u32>, usize)> = counts.iter().map(|(c, n)| (c, *n)).collect();
    Bm25::new(&refs, &q, k1, b).get_scores(&q)
}
