use std::path::PathBuf;

fn main() {
    let stub_check = std::env::var_os("CARGO_FEATURE_STUB_CHECK").is_some();
    let generate_docs = std::env::var_os("CARGO_FEATURE_GENERATE_DOCS").is_some();
    if stub_check || generate_docs {
        println!("cargo:rerun-if-changed=src");
        println!("cargo:rerun-if-changed=pyochain");

        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        pyochain_build::run(&root, stub_check, generate_docs);
    }
}
