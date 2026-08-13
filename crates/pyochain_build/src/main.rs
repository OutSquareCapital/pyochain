use std::path::PathBuf;

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    match std::env::args().nth(1).as_deref() {
        Some("generate-docs") => pyochain_build::run(&root, false, true),
        Some("stub-check") => pyochain_build::run(&root, true, false),
        Some("all") => pyochain_build::run(&root, true, true),
        _ => {
            eprintln!("Usage: cargo run -p pyochain-build -- <generate-docs|stub-check|all>");
            std::process::exit(2);
        }
    }
}
