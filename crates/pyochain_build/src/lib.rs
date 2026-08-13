use std::path::Path;

use anstream::ColorChoice;

mod generate_docs;
mod parse;
mod stub_check;

pub fn run(root: &Path, stub_check_task: bool, generate_docs_task: bool) {
    ColorChoice::Always.write_global();

    let source_root = root.join("src");
    let stub_root = root.join("pyochain");
    let pyclasses = parse::get_pyclasses(&source_root, &source_root.join("lib.rs"));
    if stub_check_task {
        stub_check::run(&stub_root, &pyclasses);
    }
    if generate_docs_task {
        generate_docs::run(root, &pyclasses);
    }
}
