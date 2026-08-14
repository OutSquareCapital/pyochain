mod check_nav;
mod generate_docs;
mod parse;
mod paths;
mod stub_check;
use std::path::PathBuf;

use tap::Pipe;

pub fn run(root: PathBuf) {
    let root = paths::Root::new(root);
    let stub_root = root.join("pyochain").pipe(paths::Root::new);
    let pyclasses = root
        .join("src")
        .pipe(paths::Root::new)
        .pipe(|r| parse::get_pyclasses(&r, "lib.rs"));
    stub_check::run(&stub_root, &pyclasses);
    generate_docs::run(&root, &pyclasses);
    check_nav::run(&root);
}
