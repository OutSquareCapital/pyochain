use std::path::PathBuf;

fn main() {
    if std::env::var_os("CARGO_FEATURE_BUILD_DOCS").is_some() {
        println!("cargo:rerun-if-changed=src");
        println!("cargo:rerun-if-changed=pyochain");
        pyochain_build::run(PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    }
}
