mod generate_docs;
mod parse;
mod paths;
mod stub_check;
mod write;
use owo_colors::OwoColorize;
use std::path::PathBuf;

use tap::prelude::*;
pub fn run(root: PathBuf) {
    let root = paths::Root::new(root);
    let stubs = root.join("pyochain").pipe(paths::Root::new);
    let docs = root.join("docs").pipe(paths::Root::new);
    let src = root.join("src").pipe(paths::Root::new);
    parse::get_pyclasses(&src, "lib.rs")
        .pipe(|pyclasses| stub_check::run(&stubs, &pyclasses))
        .pipe(|pyclasses| generate_docs::run(&root, docs, pyclasses))
        .pipe(|result| match result {
            Ok(msg) => msg.green().bold().to_string().pipe(|msg| println!("{msg}")),
            Err(err) => err.red().bold().to_string().pipe(|err| println!("{err}")),
        });
}
