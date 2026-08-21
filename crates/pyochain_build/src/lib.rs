mod generate_docs;
mod parse;
mod paths;
use std::path::PathBuf;

pub fn run(root: PathBuf) {
    let stubs = paths::Related::new(root.join("pyochain"), "pyochain");
    let docs = paths::Related::new(root.join("docs"), "reference");
    let src = paths::Related::new(root.join("src"), "lib.rs");
    let zensical = paths::Related::new(root, "zensical.toml");
    generate_docs::run(zensical, src, docs, stubs)
        .expect("Failed to generate documentation due to an unexpected error.");
}
