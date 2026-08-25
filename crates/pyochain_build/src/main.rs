use anstream::ColorChoice;
use std::{path::PathBuf, process::ExitCode};
use tap::Pipe;

fn main() -> ExitCode {
    ColorChoice::Always.write_global();
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .pipe(pyochain_build::run)
        .map_or_else(
            |error| {
                anstream::eprintln!("Documentation generation failed: {error}");
                ExitCode::FAILURE
            },
            |()| ExitCode::SUCCESS,
        )
}
