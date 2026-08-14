use anstream::ColorChoice;
use std::path::PathBuf;
use tap::Pipe;

fn main() {
    ColorChoice::Always.write_global();

    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .pipe(pyochain_build::run)
}
