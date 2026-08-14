mod check_nav;
mod generate_docs;
mod parse;
mod paths;
mod stub_check;
mod write;
use std::path::PathBuf;

use tap::prelude::*;
pub fn run(root: PathBuf) {
    let root = paths::Root::new(root);
    let stubs = root.join("pyochain").pipe(paths::Root::new);
    let docs = root.join("docs").pipe(paths::Root::new);
    let src = root.join("src").pipe(paths::Root::new);
    parse::get_pyclasses(&src, "lib.rs")
        .tap(|pyclasses| stub_check::run(&stubs, pyclasses))
        .pipe_ref(|pyclasses| generate_docs::run(docs, pyclasses))
        .pipe(|docs_paths| check_nav::run(root, docs_paths));
}
