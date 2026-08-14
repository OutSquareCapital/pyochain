use std::path::PathBuf;

use crate::parse::Root;

mod generate_docs;
mod parse;
mod stub_check;
pub fn run(root: PathBuf) {
    let root = Root::new(root);
    let source_root = Root::new(root.join("src"));
    let stub_root = Root::new(root.join("pyochain"));
    let pyclasses = source_root.get_pyclasses("lib.rs");
    let gen_doc = || {
        generate_docs::run(&root, &pyclasses);
    };
    let stub_check = || {
        stub_check::run(&stub_root, &pyclasses);
    };
    match std::env::args().nth(1).as_deref() {
        Some("generate-docs") => gen_doc(),
        Some("stub-check") => stub_check(),
        Some("all") => {
            gen_doc();
            stub_check();
        }
        _ => {
            eprintln!("Usage: cargo run -p pyochain-build -- <generate-docs|stub-check|all>");
            std::process::exit(2);
        }
    }
}
pub fn run_all(root: PathBuf) {
    let root = Root::new(root);
    let source_root = Root::new(root.join("src"));
    let stub_root = Root::new(root.join("pyochain"));
    let pyclasses = source_root.get_pyclasses("lib.rs");
    generate_docs::run(&root, &pyclasses);
    stub_check::run(&stub_root, &pyclasses);
}
