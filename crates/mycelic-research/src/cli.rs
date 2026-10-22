//! A small `--option value` parser for the subcommands that used argparse.

use anyhow::{Result, bail};

/// Parsed `--name value` options (repeatable) and positional arguments.
#[derive(Debug, Default)]
pub struct Args {
    pub opts: Vec<(String, String)>,
    pub positional: Vec<String>,
}

impl Args {
    pub fn parse(args: &[String]) -> Result<Args> {
        let mut out = Args::default();
        let mut it = args.iter();
        while let Some(a) = it.next() {
            if let Some(name) = a.strip_prefix("--") {
                let (name, value) = match name.split_once('=') {
                    Some((n, v)) => (n.to_string(), v.to_string()),
                    None => match it.next() {
                        Some(v) => (name.to_string(), v.clone()),
                        None => bail!("--{name} needs a value"),
                    },
                };
                out.opts.push((name.replace('-', "_"), value));
            } else {
                out.positional.push(a.clone());
            }
        }
        Ok(out)
    }

    pub fn get(&self, name: &str) -> Option<&str> {
        self.opts.iter().rev().find(|o| o.0 == name).map(|o| o.1.as_str())
    }

    pub fn all(&self, name: &str) -> Vec<&str> {
        self.opts.iter().filter(|o| o.0 == name).map(|o| o.1.as_str()).collect()
    }

    pub fn required(&self, name: &str) -> Result<&str> {
        match self.get(name) {
            Some(v) => Ok(v),
            None => bail!("the following arguments are required: --{}", name.replace('_', "-")),
        }
    }
}

/// "0,1,2" -> [0, 1, 2]
pub fn list<T: std::str::FromStr>(s: &str) -> Result<Vec<T>> {
    s.split(',').map(|x| x.trim().parse::<T>().map_err(|_| anyhow::anyhow!("bad number {x:?}"))).collect()
}
