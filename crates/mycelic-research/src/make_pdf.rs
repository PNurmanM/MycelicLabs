//! Render a generated report to a paginated PDF (`make_pdf.py`).
//!
//! The report is a long Markdown document with a lot of wide numeric tables and nine figures, which is exactly the
//! shape that converts badly by default: the tables overflow the page and the figures land wherever they fall.  So
//! this does three specific things rather than a generic Markdown-to-PDF pass:
//!
//! * sizes each table's type from its column count, so a 13-column baseline table fits the page instead of being
//!   clipped,
//! * keeps rows, figures and short sections off page boundaries, and repeats table headers when a long table does
//!   have to break,
//! * forces the collapsed contents block open, since a `<details>` element prints as just its summary line.
//!
//! Markdown is converted with pulldown-cmark (CommonMark + tables), adjusted towards what Python-Markdown with
//! `tables, fenced_code, attr_list, md_in_html` produced: text escaped the same way (`&`, `<`, `>` only), the
//! contents `<details>` opened the same way, and table alignment as `style="text-align: ..."` (which the CSS keys
//! on).  Chrome or Chromium prints the HTML through the DevTools protocol's `Page.printToPDF`, with the same page
//! options the Python passed through Playwright (A4, backgrounds, header/footer templates with page numbers,
//! margins).  The browser is found at run time: `MYCELIC_CHROME`, then the usual install locations (Google Chrome,
//! Chromium, Playwright's browser caches, `/opt/pw-browsers`), then `PATH`.
//!
//! ```text
//! mycelic-research make_pdf [src out [title [subtitle]]]
//! ```

use crate::runner::repo_root;
use anyhow::{Context, Result, anyhow, bail};
use base64::Engine as _;
use pulldown_cmark::{CowStr, Event, Options, Parser, Tag, TagEnd};
use regex::{Captures, Regex};
use serde_json::{Value, json};
use std::io::ErrorKind;
use std::net::TcpStream;
use std::path::{Component, Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use tungstenite::stream::MaybeTlsStream;
use tungstenite::{Message, WebSocket};

pub const TITLE: &str = "Mycelic — hierarchical enterprise intelligence";
pub const SUBTITLE: &str = "Benchmark report and architecture recommendation";

/// Column-count thresholds for table type size.  The widest table in the report has 13 columns; at 5.6pt that
/// fits A4 portrait with margins.
pub const TABLE_SIZES: [(usize, &str); 3] = [(12, "t-xs"), (9, "t-sm"), (7, "t-md")];

/// The stylesheet, verbatim from the Python.
pub const CSS: &str = r##"
@page {
  size: A4 portrait;
  margin: 17mm 14mm 16mm 14mm;
}
:root {
  --ink: #16181d;
  --muted: #5b6270;
  --rule: #d4d8e0;
  --rule-soft: #e8ebf0;
  --accent: #2f4858;
  --band: #f4f6f9;
}
* { box-sizing: border-box; }
body {
  font-family: "Charter", "Georgia", "Times New Roman", serif;
  font-size: 9.6pt;
  line-height: 1.48;
  color: var(--ink);
  margin: 0;
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}
p { margin: 0 0 0.62em; orphans: 3; widows: 3; }
em { color: var(--muted); }

h1, h2, h3, h4 {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  color: var(--accent);
  line-height: 1.22;
  margin: 1.15em 0 0.45em;
  page-break-after: avoid;
  break-after: avoid;
}
/* PART dividers are h1 in the source; each opens a page. */
h1 {
  font-size: 15pt;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  border-bottom: 1.6pt solid var(--accent);
  padding-bottom: 0.3em;
  page-break-before: always;
  break-before: page;
}
h1.doc-title, h1:first-of-type { page-break-before: avoid; break-before: auto; }
h2 { font-size: 12.4pt; border-bottom: 0.6pt solid var(--rule); padding-bottom: 0.2em; }
h3 { font-size: 10.6pt; }
h4 { font-size: 9.8pt; color: var(--muted); }

ul, ol { margin: 0 0 0.62em; padding-left: 1.25em; }
li { margin-bottom: 0.22em; }

code, kbd {
  font-family: "SF Mono", "DejaVu Sans Mono", Menlo, Consolas, monospace;
  font-size: 0.86em;
  background: var(--band);
  padding: 0.05em 0.3em;
  border-radius: 2px;
}
pre {
  background: var(--band);
  border: 0.5pt solid var(--rule);
  border-radius: 3px;
  padding: 7pt 9pt;
  font-size: 7.2pt;
  line-height: 1.34;
  overflow: visible;
  white-space: pre-wrap;
  page-break-inside: avoid;
  break-inside: avoid;
}
pre code { background: none; padding: 0; font-size: inherit; }

table {
  border-collapse: collapse;
  width: 100%;
  margin: 0.5em 0 0.9em;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 7.6pt;
  font-variant-numeric: tabular-nums;
}
table.t-md { font-size: 7.0pt; }
table.t-sm { font-size: 6.4pt; }
table.t-xs { font-size: 5.6pt; }
thead { display: table-header-group; }
th, td {
  border-bottom: 0.5pt solid var(--rule-soft);
  padding: 2.6pt 3.4pt;
  vertical-align: top;
  word-break: normal;
  /* break-word, not anywhere: `anywhere` is taken into account when the
     browser computes a column's minimum width, which lets it squeeze a
     numeric column until "10000" wraps to "1000/0". */
  overflow-wrap: break-word;
}
/* Numbers never wrap; markdown right-aligns the numeric columns for us. */
td[style*="right"], th[style*="right"] { white-space: nowrap; }
/* Long header labels may wrap — that is what makes the numbers fit. */
th { overflow-wrap: anywhere; hyphens: auto; }
th[style*="right"] { white-space: normal; }
th {
  background: var(--band);
  border-bottom: 0.9pt solid var(--rule);
  text-align: left;
  font-weight: 600;
  color: var(--accent);
}
tr { page-break-inside: avoid; break-inside: avoid; }
td sub, td sup { color: var(--muted); font-size: 0.82em; }

img {
  display: block;
  max-width: 100%;
  height: auto;
  margin: 0.7em auto 0.4em;
  page-break-inside: avoid;
  break-inside: avoid;
}

blockquote {
  margin: 0.6em 0;
  padding-left: 0.8em;
  border-left: 2pt solid var(--rule);
  color: var(--muted);
}
hr { border: 0; border-top: 0.6pt solid var(--rule); margin: 1.1em 0; }
a { color: var(--accent); text-decoration: none; }

/* The contents block is a <details>; print it expanded and unboxed. */
details { display: block; }
details > summary { list-style: none; font-weight: 600; color: var(--accent); }
details > summary::-webkit-details-marker { display: none; }
.contents {
  border: 0.6pt solid var(--rule);
  border-radius: 3px;
  background: #fbfcfd;
  padding: 8pt 12pt 4pt;
  margin: 0.9em 0 1.1em;
  column-count: 2;
  column-gap: 16pt;
  font-size: 8.4pt;
}
.contents summary { column-span: all; margin-bottom: 4pt; }
.contents ul { padding-left: 1.0em; margin-bottom: 0.4em; }
.contents li { margin-bottom: 0.1em; }
.contents p { break-inside: avoid; margin-bottom: 0.25em; }

.cover {
  page-break-after: always;
  break-after: page;
  padding-top: 28mm;
}
.cover .t { font-size: 25pt; line-height: 1.16; margin: 0 0 6pt;
            font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            color: var(--accent); font-weight: 600; }
.cover .s { font-size: 13pt; color: var(--muted); margin: 0 0 26pt;
            font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; }
.cover .rule { border-top: 2.2pt solid var(--accent); width: 62mm;
               margin: 0 0 22pt; }
.cover dl { font-size: 9.4pt; margin: 0; }
.cover dt { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            font-size: 7.6pt; letter-spacing: 0.08em; text-transform: uppercase;
            color: var(--muted); margin-top: 11pt; }
.cover dd { margin: 1pt 0 0; }
.cover .note { margin-top: 30mm; font-size: 8.6pt; color: var(--muted);
               border-top: 0.6pt solid var(--rule); padding-top: 8pt; }
"##;

/// The running header (`__TITLE__`/`__SUBTITLE__` are filled in), verbatim from the Python.
pub const HEADER: &str = r##"
<div style="font-family:Helvetica,Arial,sans-serif;font-size:6.5pt;color:#8b919c;
            width:100%;padding:0 14mm;display:flex;justify-content:space-between;">
  <span>__TITLE__</span>
  <span>__SUBTITLE__</span>
</div>
"##;

/// The running footer with page numbers, verbatim from the Python.
pub const FOOTER: &str = r##"
<div style="font-family:Helvetica,Arial,sans-serif;font-size:6.5pt;color:#8b919c;
            width:100%;padding:0 14mm;display:flex;justify-content:space-between;">
  <span>Generated from research/mycelic/artifacts — every figure is a real run</span>
  <span><span class="pageNumber"></span> / <span class="totalPages"></span></span>
</div>
"##;

/// `docs/MYCELIC_ENTERPRISE.md` and its PDF: what `make_pdf` renders without arguments.
pub fn default_paths() -> (PathBuf, PathBuf) {
    let docs = repo_root().join("docs");
    (docs.join("MYCELIC_ENTERPRISE.md"), docs.join("MYCELIC_ENTERPRISE.pdf"))
}

// -------------------------------------------------------------------------------------------------------------
// Markdown -> HTML
// -------------------------------------------------------------------------------------------------------------

/// Python-Markdown's text escaping: `&`, `<` and `>` only (quotes stay as they are).
fn escape_text(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

/// `html.escape(s)` (quote=True).
pub fn html_escape(s: &str) -> String {
    escape_text(s).replace('"', "&quot;").replace('\'', "&#x27;")
}

/// Markdown to an HTML fragment: CommonMark with tables, fenced code and raw HTML, text escaped as
/// Python-Markdown escapes it.
pub fn markdown_to_html(md: &str) -> String {
    let parser = Parser::new_ext(md, Options::ENABLE_TABLES);
    let mut in_code_block = false;
    let events = parser.map(|ev| match ev {
        Event::Start(Tag::CodeBlock(kind)) => {
            in_code_block = true;
            Event::Start(Tag::CodeBlock(kind))
        }
        Event::End(TagEnd::CodeBlock) => {
            in_code_block = false;
            Event::End(TagEnd::CodeBlock)
        }
        Event::Text(t) => Event::Html(CowStr::from(escape_text(&t))),
        Event::Code(c) if !in_code_block => Event::Html(CowStr::from(format!("<code>{}</code>", escape_text(&c)))),
        other => other,
    });
    let mut out = String::new();
    pulldown_cmark::html::push_html(&mut out, events);
    out
}

/// `_size_tables`: give each table a class chosen from its column count.
///
/// A 13-column table at the body table size runs off the page; shrinking every table to fit the widest one would
/// make the narrow ones unreadable.
pub fn size_tables(doc: &str) -> String {
    let table = Regex::new(r"(?s)<table>.*?</table>").expect("regex");
    let thead = Regex::new(r"(?s)<thead>.*?</thead>").expect("regex");
    let th = Regex::new(r"(?s)<th[^>]*>(.*?)</th>").expect("regex");
    let tag = Regex::new(r"<[^>]+>").expect("regex");
    table
        .replace_all(doc, |m: &Captures| {
            let block = &m[0];
            let Some(head) = thead.find(block) else { return block.to_string() };
            let cells: Vec<&str> =
                th.captures_iter(head.as_str()).map(|c| c.get(1).map_or("", |g| g.as_str())).collect();
            let n = cells.len();
            // A table with few but very wordy headers ("cited evidence names the claimed entity") needs the same
            // treatment as a table with many narrow ones, so count each long header as extra columns.
            let longest = cells.iter().map(|c| tag.replace_all(c, "").chars().count()).max().unwrap_or(0);
            let weight = n + longest.saturating_sub(14) / 9;
            for (threshold, class) in TABLE_SIZES {
                if weight >= threshold {
                    return block.replacen("<table>", &format!("<table class=\"{class}\">"), 1);
                }
            }
            block.to_string()
        })
        .into_owned()
}

/// `os.path.normpath` for an absolute path: `.` and `..` resolved lexically.
fn normpath(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other),
        }
    }
    out
}

/// `_absolutise_images`: rewrite the document-relative figure paths to absolute file URLs.
pub fn absolutise_images(doc: &str, docs_dir: &Path) -> String {
    let img = Regex::new(r#"<img[^>]*src="([^"]+)""#).expect("regex");
    img.replace_all(doc, |m: &Captures| {
        let src = &m[1];
        if ["http:", "https:", "file:", "data:"].iter().any(|p| src.starts_with(p)) {
            return m[0].to_string();
        }
        let p = normpath(&docs_dir.join(src));
        m[0].replace(src, &format!("file://{}", p.display()))
    })
    .into_owned()
}

/// `f"{n:,}"`
fn thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// `_cover`: a title page carrying the facts a reader needs before page one.
pub fn cover(md_text: &str, title: &str, subtitle: &str, today: &str) -> String {
    // Take the scales from the scale-trend table's header, which is the one place every measured enterprise size
    // appears; the per-scale section headings only cover the scales the full head-to-head was run at.
    let header = Regex::new(r"\n\| architecture \|((?: [\d,]+ \|)+)\n").expect("regex");
    let number = Regex::new(r"[\d,]+").expect("regex");
    let parse = |s: &str| s.replace(',', "").parse::<u64>().ok();
    let scales: Vec<u64> = match header.captures(md_text) {
        Some(h) => {
            let mut v: Vec<u64> = number.find_iter(&h[1]).filter_map(|m| parse(m.as_str())).collect();
            v.sort();
            v
        }
        None => {
            // sorted(set(...))
            let heading = Regex::new(r"#### ([\d,]+) users").expect("regex");
            let set: std::collections::BTreeSet<u64> =
                heading.captures_iter(md_text).filter_map(|c| parse(&c[1])).collect();
            set.into_iter().collect()
        }
    };
    let n_tables = md_text.matches("\n|---").count();
    let n_figs = md_text.matches("![").count();
    let n_sections = Regex::new(r"(?m)^## \d+\.").expect("regex").find_iter(md_text).count();
    let scale_txt = if scales.is_empty() {
        "—".to_string()
    } else {
        scales.iter().map(|&s| thousands(s)).collect::<Vec<_>>().join(", ")
    };
    let short_title = title.split(" — ").next().unwrap_or(title);
    format!(
        r#"
<section class="cover">
  <div class="t">{}</div>
  <div class="s">{}</div>
  <div class="rule"></div>
  <dl>
    <dt>Question</dt>
    <dd>How should thousands of private user-level agents abstract, route,
        combine, question, verify and propagate what they know upward, so that
        an enterprise kernel discovers what no individual could?</dd>
    <dt>Enterprise sizes measured</dt><dd>{scale_txt} users</dd>
    <dt>Contents</dt>
    <dd>{n_sections} sections · {n_tables} tables · {n_figs} figures</dd>
    <dt>Generated</dt><dd>{today}</dd>
  </dl>
  <div class="note">
    Every table and figure in this document is computed from the raw per-run
    records in <code>research/mycelic/artifacts/</code> at generation time.
    No number is transcribed by hand. Section&nbsp;5 states exactly which
    results were measured on real models and which were simulated.
  </div>
</section>
"#,
        html_escape(short_title),
        html_escape(subtitle)
    )
}

/// `build_html()`: the whole printable document for the Markdown source `src`.
pub fn build_html(src: &Path, md_text: &str, title: &str, subtitle: &str, today: &str) -> String {
    // Two things have to happen before conversion, not after.  A <details> prints as just its summary line, so it
    // is opened here (with the attributes Python-Markdown's md_in_html wrote); Markdown inside it is converted
    // because CommonMark resumes Markdown after the HTML block's blank line.
    let md_html = md_text.replacen("<details>", "<details class=\"contents\" open=\"open\">", 1);
    let body = markdown_to_html(&md_html);
    let body = size_tables(&body);
    let docs_dir = src.parent().unwrap_or(Path::new("."));
    let body = absolutise_images(&body, docs_dir);
    // The source's own title block is replaced by the cover page.
    let title_block = Regex::new(r"(?s)\A<h1>.*?</h1>\s*<h2>.*?</h2>").expect("regex");
    let body = title_block.replacen(&body, 1, "");
    format!(
        "<!doctype html><html><head><meta charset='utf-8'><title>{}</title><style>{CSS}</style></head><body>{}{body}</body></html>",
        html_escape(title),
        cover(md_text, title, subtitle, today)
    )
}

/// The header template as the Python filled it: the title, and the subtitle up to its first comma or " and ",
/// at most 48 characters.
pub fn header_template(title: &str, subtitle: &str) -> String {
    let short = subtitle.split(',').next().unwrap_or("").split(" and ").next().unwrap_or("");
    let short: String = short.chars().take(48).collect();
    HEADER.replace("__TITLE__", title).replace("__SUBTITLE__", &short)
}

// -------------------------------------------------------------------------------------------------------------
// Printing through Chrome's DevTools protocol
// -------------------------------------------------------------------------------------------------------------

/// Chrome / Chromium executables to try, most specific first.
fn browser_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(p) = std::env::var_os("MYCELIC_CHROME") {
        out.push(PathBuf::from(p));
    }
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default();
    for p in [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
        "/opt/pw-browsers/chromium/chrome-linux/chrome",
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
    ] {
        out.push(PathBuf::from(p));
    }
    // Playwright's caches (newest revision first): full Chromium builds, then the headless shell
    let caches = [
        home.join("Library/Caches/ms-playwright"),
        home.join(".cache/ms-playwright"),
        PathBuf::from("/opt/pw-browsers"),
    ];
    let inner = [
        "chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
        "chrome-mac/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
        "chrome-mac/Chromium.app/Contents/MacOS/Chromium",
        "chrome-linux64/chrome",
        "chrome-linux/chrome",
        "chrome-headless-shell-mac-arm64/chrome-headless-shell",
        "chrome-headless-shell-mac-x64/chrome-headless-shell",
        "chrome-headless-shell-linux64/chrome-headless-shell",
    ];
    for cache in caches {
        let Ok(entries) = std::fs::read_dir(&cache) else { continue };
        let mut dirs: Vec<(u32, bool, PathBuf)> = entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter_map(|p| {
                let name = p.file_name()?.to_str()?.to_string();
                let (kind, rev) = name.rsplit_once('-')?;
                let shell = match kind {
                    "chromium" => false,
                    "chromium_headless_shell" => true,
                    _ => return None,
                };
                Some((rev.parse().ok()?, shell, p))
            })
            .collect();
        // full builds before headless shells, newest first within each
        dirs.sort_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)));
        for (_, _, dir) in dirs {
            out.extend(inner.iter().map(|i| dir.join(i)));
        }
    }
    let path_var = std::env::var_os("PATH").unwrap_or_default();
    for dir in std::env::split_paths(&path_var) {
        for name in ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome"] {
            out.push(dir.join(name));
        }
    }
    out
}

/// The first browser executable that exists (`MYCELIC_CHROME` must exist if set).
pub fn find_browser() -> Result<PathBuf> {
    if let Some(p) = std::env::var_os("MYCELIC_CHROME") {
        let p = PathBuf::from(p);
        if !p.is_file() {
            bail!("MYCELIC_CHROME={} is not a file", p.display());
        }
        return Ok(p);
    }
    browser_candidates().into_iter().find(|p| p.is_file()).ok_or_else(|| {
        anyhow!(
            "no Chrome/Chromium found: set MYCELIC_CHROME to the browser executable (or install Google Chrome, \
             Chromium, or Playwright's chromium)"
        )
    })
}

/// Page options for `Page.printToPDF`, as Playwright sent them for `format="A4"` and the Python's margins.
pub struct PrintOptions {
    pub header_template: String,
    pub footer_template: String,
}

/// A headless browser and its DevTools websocket; the browser is closed and its profile removed on drop.
struct Browser {
    child: Child,
    profile: PathBuf,
    ws: WebSocket<MaybeTlsStream<TcpStream>>,
    next_id: u64,
}

impl Browser {
    fn launch(exe: &Path) -> Result<Browser> {
        let profile = std::env::temp_dir().join(format!(
            "mycelic-pdf-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos())
        ));
        std::fs::create_dir_all(&profile)?;
        let child = Command::new(exe)
            .args([
                "--headless",
                "--remote-debugging-port=0",
                "--no-sandbox",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-gpu",
                "--disable-extensions",
                "--hide-scrollbars",
                "--mute-audio",
            ])
            .arg(format!("--user-data-dir={}", profile.display()))
            .arg("about:blank")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| format!("starting {}", exe.display()))?;
        // the browser writes its port and browser target path here once it listens
        let port_file = profile.join("DevToolsActivePort");
        let start = Instant::now();
        let (port, path) = loop {
            if let Ok(text) = std::fs::read_to_string(&port_file) {
                let mut lines = text.lines();
                if let (Some(port), Some(path)) = (lines.next(), lines.next()) {
                    break (port.trim().to_string(), path.trim().to_string());
                }
            }
            if start.elapsed() > Duration::from_secs(30) {
                let mut child = child;
                let _ = child.kill();
                bail!("{} did not open its DevTools port within 30 s", exe.display());
            }
            std::thread::sleep(Duration::from_millis(50));
        };
        let url = format!("ws://127.0.0.1:{port}{path}");
        let (ws, _) = tungstenite::connect(url.as_str()).with_context(|| format!("connecting to {url}"))?;
        if let MaybeTlsStream::Plain(s) = ws.get_ref() {
            s.set_read_timeout(Some(Duration::from_secs(120)))?;
        }
        Ok(Browser { child, profile, ws, next_id: 0 })
    }

    /// Send a command (to a page session if given) and wait for its result; events met on the way go to `on_event`.
    fn call(
        &mut self,
        session: Option<&str>,
        method: &str,
        params: Value,
        on_event: &mut dyn FnMut(&Value),
    ) -> Result<Value> {
        self.next_id += 1;
        let id = self.next_id;
        let mut msg = json!({"id": id, "method": method, "params": params});
        if let Some(s) = session {
            msg["sessionId"] = json!(s);
        }
        self.ws.send(Message::text(msg.to_string()))?;
        loop {
            let reply: Value = match self.ws.read() {
                Ok(Message::Text(t)) => serde_json::from_str(t.as_str())?,
                Ok(_) => continue,
                Err(tungstenite::Error::Io(e))
                    if e.kind() == ErrorKind::WouldBlock || e.kind() == ErrorKind::TimedOut =>
                {
                    bail!("{method}: the browser did not answer in time")
                }
                Err(e) => return Err(e.into()),
            };
            if reply.get("id").and_then(Value::as_u64) == Some(id) {
                if let Some(err) = reply.get("error") {
                    bail!("{method}: {err}");
                }
                return Ok(reply.get("result").cloned().unwrap_or(Value::Null));
            }
            on_event(&reply);
        }
    }

    /// Wait for an event on a session.
    fn wait_event(&mut self, session: &str, method: &str) -> Result<()> {
        loop {
            let ev: Value = match self.ws.read()? {
                Message::Text(t) => serde_json::from_str(t.as_str())?,
                _ => continue,
            };
            if ev["method"] == method && ev["sessionId"] == session {
                return Ok(());
            }
        }
    }
}

impl Drop for Browser {
    fn drop(&mut self) {
        let _ = self.ws.send(Message::text(json!({"id": 0, "method": "Browser.close"}).to_string()));
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            if let Ok(Some(_)) = self.child.try_wait() {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.profile);
    }
}

/// Print an HTML file to PDF with headless Chrome: A4, backgrounds, header and footer, the Python's margins.
pub fn print_pdf(browser_exe: &Path, html_path: &Path, pdf_path: &Path, opts: &PrintOptions) -> Result<()> {
    let mut b = Browser::launch(browser_exe)?;
    let mut ignore = |_: &Value| {};
    let target = b.call(None, "Target.createTarget", json!({"url": "about:blank"}), &mut ignore)?;
    let target_id = target["targetId"].as_str().context("createTarget: no targetId")?.to_string();
    let attached =
        b.call(None, "Target.attachToTarget", json!({"targetId": target_id, "flatten": true}), &mut ignore)?;
    let session = attached["sessionId"].as_str().context("attachToTarget: no sessionId")?.to_string();
    let s = Some(session.as_str());
    b.call(s, "Page.enable", json!({}), &mut ignore)?;
    let mut loaded = false;
    let url = format!("file://{}", html_path.display());
    b.call(s, "Page.navigate", json!({"url": url}), &mut |ev| {
        loaded |= ev["method"] == "Page.loadEventFired";
    })?;
    if !loaded {
        b.wait_event(&session, "Page.loadEventFired")?;
    }
    // Playwright's format="A4" is 8.27 x 11.7 in; margins 17/16/14/14 mm as the Python gave them
    let mm = |v: f64| v / 25.4;
    let params = json!({
        "landscape": false,
        "displayHeaderFooter": true,
        "headerTemplate": opts.header_template,
        "footerTemplate": opts.footer_template,
        "printBackground": true,
        "scale": 1.0,
        "paperWidth": 8.27,
        "paperHeight": 11.7,
        "marginTop": mm(17.0),
        "marginBottom": mm(16.0),
        "marginLeft": mm(14.0),
        "marginRight": mm(14.0),
        "pageRanges": "",
        "preferCSSPageSize": false,
        "generateTaggedPDF": false,
        "generateDocumentOutline": false,
        "transferMode": "ReturnAsStream",
    });
    let printed = b.call(s, "Page.printToPDF", params, &mut ignore)?;
    let handle = printed["stream"].as_str().context("printToPDF: no stream")?.to_string();
    let mut pdf = Vec::new();
    loop {
        let chunk = b.call(s, "IO.read", json!({"handle": handle, "size": 1 << 20}), &mut ignore)?;
        let data = chunk["data"].as_str().unwrap_or("");
        if chunk["base64Encoded"].as_bool().unwrap_or(false) {
            pdf.extend(base64::engine::general_purpose::STANDARD.decode(data).context("decoding the PDF")?);
        } else {
            pdf.extend_from_slice(data.as_bytes());
        }
        if chunk["eof"].as_bool().unwrap_or(true) {
            break;
        }
    }
    b.call(s, "IO.close", json!({"handle": handle}), &mut ignore)?;
    std::fs::write(pdf_path, pdf).with_context(|| format!("writing {}", pdf_path.display()))
}

/// `build(src, out, title, subtitle)`: the HTML written next to the source (so its relative paths still resolve),
/// printed to `out`, and removed again.
pub fn build(src: &Path, out: &Path, title: Option<&str>, subtitle: Option<&str>) -> Result<PathBuf> {
    let src = std::path::absolute(src)?;
    let out = std::path::absolute(out)?;
    let title = title.filter(|t| !t.is_empty()).unwrap_or(TITLE);
    let subtitle = subtitle.filter(|t| !t.is_empty()).unwrap_or(SUBTITLE);
    let md_text = std::fs::read_to_string(&src).with_context(|| format!("reading {}", src.display()))?;
    let today = chrono::Local::now().date_naive().format("%Y-%m-%d").to_string();
    let doc = build_html(&src, &md_text, title, subtitle, &today);
    let browser = find_browser()?;
    let tmp = src.parent().unwrap_or(Path::new(".")).join(".report.tmp.html");
    std::fs::write(&tmp, doc)?;
    let opts = PrintOptions { header_template: header_template(title, subtitle), footer_template: FOOTER.to_string() };
    let printed = print_pdf(&browser, &tmp, &out, &opts);
    let _ = std::fs::remove_file(&tmp);
    printed?;
    Ok(out)
}

/// `python -m research.mycelic.make_pdf [src out [title [subtitle]]]`
pub fn main(args: &[String]) -> Result<()> {
    let p = if args.len() >= 2 {
        build(
            Path::new(&args[0]),
            Path::new(&args[1]),
            args.get(2).map(String::as_str),
            args.get(3).map(String::as_str),
        )?
    } else {
        let (src, out) = default_paths();
        build(&src, &out, None, None)?
    };
    let size = std::fs::metadata(&p)?.len() as f64 / 1e6;
    println!("wrote {} {size:.2} MB", p.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_are_sized_by_column_count() {
        let row = |n: usize| (0..n).map(|i| format!("| c{i} ")).collect::<String>() + "|\n";
        let sep = |n: usize| (0..n).map(|_| "|---").collect::<String>() + "|\n";
        let md = format!("{}{}{}\n\n{}{}{}", row(13), sep(13), row(13), row(3), sep(3), row(3));
        let html = size_tables(&markdown_to_html(&md));
        assert!(html.contains("<table class=\"t-xs\">"), "{html}");
        assert_eq!(html.matches("<table>").count(), 1, "the 3-column table keeps the body size");
        // one long header counts as extra columns: 3 + (41 - 14) / 9 = 6 -> still body size; 50 chars -> 7
        let wide = format!("| a | b | {} |\n|---|---|---|\n| 1 | 2 | 3 |\n", "x".repeat(50));
        assert!(size_tables(&markdown_to_html(&wide)).contains("<table class=\"t-md\">"));
    }

    #[test]
    fn alignment_is_a_style_attribute_the_css_can_match() {
        let html = markdown_to_html("| a | b |\n|---|--:|\n| x | 1 |\n");
        assert!(html.contains("<td style=\"text-align: right\">1</td>"), "{html}");
    }

    #[test]
    fn text_is_escaped_like_python_markdown() {
        assert_eq!(markdown_to_html("a \"b\" & c's <d>\n"), "<p>a \"b\" &amp; c's <d></p>\n");
        assert_eq!(markdown_to_html("`x < \"y\"`\n"), "<p><code>x &lt; \"y\"</code></p>\n");
        assert_eq!(markdown_to_html("```\n<a> & b\n```\n"), "<pre><code>&lt;a&gt; &amp; b\n</code></pre>\n");
    }

    #[test]
    fn contents_block_is_opened_and_its_markdown_converted() {
        let md =
            "# T\n\n## S\n\nintro\n\n<details>\n<summary><b>Contents</b></summary>\n\n\n* [one](#one)\n\n</details>\n";
        let html = build_html(Path::new("/r/docs/x.md"), md, "Title — long", "Sub", "2026-01-01");
        assert!(
            html.contains("<details class=\"contents\" open=\"open\">\n<summary><b>Contents</b></summary>"),
            "{html}"
        );
        assert!(html.contains("<li><a href=\"#one\">one</a></li>"), "{html}");
        // the source's title block gives way to the cover
        assert!(!html.contains("<h1>T</h1>") && html.contains("<div class=\"t\">Title</div>"));
    }

    #[test]
    fn images_become_absolute_file_urls() {
        let html = markdown_to_html("![fig](../research/a/b.png) ![w](https://x/y.png)\n");
        let abs = absolutise_images(&html, Path::new("/repo/docs"));
        assert!(abs.contains("src=\"file:///repo/research/a/b.png\""), "{abs}");
        assert!(abs.contains("src=\"https://x/y.png\""), "{abs}");
    }

    #[test]
    fn cover_counts_and_scales() {
        let md = "x\n| architecture | 2,000 | 10,000 | 100,000 |\n|---|---|---|---|\n## 1. a\n## 2. b\n![f](p.png)\n";
        let c = cover(md, "A — b", "S & T", "2026-10-21");
        assert!(c.contains("<dd>2,000, 10,000, 100,000 users</dd>"), "{c}");
        assert!(c.contains("<dd>2 sections · 1 tables · 1 figures</dd>"), "{c}");
        assert!(c.contains("<div class=\"t\">A</div>") && c.contains("<div class=\"s\">S &amp; T</div>"));
    }

    #[test]
    fn header_subtitle_is_shortened_as_in_python() {
        let h = header_template("Mycelic vNext", "Research report, architecture, experiment matrix");
        assert!(h.contains("<span>Mycelic vNext</span>") && h.contains("<span>Research report</span>"), "{h}");
        let h = header_template("T", SUBTITLE);
        assert!(h.contains("<span>Benchmark report</span>"), "{h}");
    }
}
