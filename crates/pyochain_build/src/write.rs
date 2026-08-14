use owo_colors::OwoColorize;
use std::{
    fs,
    path::{Display, Path},
};
pub(super) enum Kind {
    New,
    Unchanged,
    Updated,
}
impl Kind {
    pub fn new(path: &Path, new_content: &str) -> Self {
        if path.exists() {
            if &fs::read_to_string(&path).unwrap() != new_content {
                Self::Updated
            } else {
                Self::Unchanged
            }
        } else {
            Self::New
        }
    }
    fn message(&self, display: Display<'_>) -> String {
        match self {
            Self::New => format!("Generated {} (new file)", display),
            Self::Unchanged => format!("Skipping {} (no changes)", display),
            Self::Updated => format!("Updating {} (content changed)", display),
        }
    }
    pub fn maybe_write(self, path: &Path, new_content: String) {
        let display = path.display();
        match self {
            Self::New | Self::Updated => {
                anstream::eprintln!("{}", self.message(display).cyan().bold());
                fs::write(&path, new_content).unwrap();
            }
            Self::Unchanged => {
                anstream::eprintln!("{}", self.message(display).cyan().bold());
            }
        }
    }
}
pub fn get_new_content(name: &str, full_path: &str) -> String {
    format!(
        "# {name}

::: {full_path}
"
    )
}
