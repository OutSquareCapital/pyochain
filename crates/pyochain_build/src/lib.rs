mod check_docstrings;
mod generate_docs;
mod parse;
mod paths;
use std::error::Error;
use std::path::PathBuf;

pub fn run(root: PathBuf) -> Result<(), Box<dyn Error>> {
    let stubs = paths::Related::new(root.join("pyochain"), "pyochain", "pyi");
    let docs = paths::Related::new(root.join("docs"), "reference", "md");
    let src = paths::Related::new(root.join("src"), "lib.rs", "rs");
    let zensical = paths::Related::new(root, "zensical.toml", "toml");
    generate_docs::run(&zensical, &src, &docs, &stubs)?;
    check_docstrings::run(&stubs)
}
