//! The figures render from the committed rows at matplotlib's pixel sizes (figsize x dpi 150).  Rendering needs a
//! font at run time; on a machine without any, the test says so and passes.

use mycelic_research::plots;
use mycelic_research::plotting::Fonts;
use mycelic_research::runner::repo_root;

/// (width, height) from a PNG's IHDR chunk.
fn png_size(bytes: &[u8]) -> (u32, u32) {
    assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n", "not a PNG");
    let w = u32::from_be_bytes(bytes[16..20].try_into().unwrap());
    let h = u32::from_be_bytes(bytes[20..24].try_into().unwrap());
    (w, h)
}

#[test]
fn every_figure_renders_at_its_matplotlib_size() {
    let fonts = match Fonts::load() {
        Ok(f) => f,
        Err(e) => {
            eprintln!("skipping: {e:#}");
            return;
        }
    };
    let art = repo_root().join("research").join("mycelic").join("artifacts");
    let out = tempfile::tempdir().unwrap();
    let written = plots::build(&art, out.path(), &fonts).unwrap();
    let expected = [
        ("scale.png", (1650, 630)),
        ("cost_quality.png", (1890, 630)),
        ("level_marginal.png", (1080, 630)),
        ("q_sweep.png", (960, 630)),
        ("discrimination.png", (1110, 660)),
        ("ablations.png", (1140, 0)),
        ("privacy.png", (1080, 660)),
        ("adversarial.png", (1950, 0)),
        ("fanin.png", (1650, 630)),
        ("old_new.png", (2160, 570)),
    ];
    assert_eq!(written.len(), expected.len(), "{written:?}");
    for (name, (w, h)) in expected {
        let bytes = std::fs::read(out.path().join(name)).unwrap();
        let (gw, gh) = png_size(&bytes);
        assert_eq!(gw, w, "{name} width");
        // the ablation and adversarial heights follow the number of rows, as in the Python
        if h > 0 {
            assert_eq!(gh, h, "{name} height");
        }
        // the same pixel size as the committed matplotlib figure
        let committed = std::fs::read(art.join("figures").join(name)).unwrap();
        assert_eq!((gw, gh), png_size(&committed), "{name} vs the committed figure");
    }
}
