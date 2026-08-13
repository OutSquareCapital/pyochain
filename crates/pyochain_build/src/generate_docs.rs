use std::fs;
use std::path::{Display, Path};

use crate::parse::PyClass;
use owo_colors::OwoColorize;

pub(super) fn run(root: &Path, pyclasses: &[PyClass]) {
    anstream::eprintln!("{}", "Generating pyochain documentation...".cyan().bold());
    let docs_ref = root.join("docs").join("reference");
    fs::create_dir_all(&docs_ref).unwrap();
    pyclasses
        .iter()
        .for_each(|pyclass| handle_class(&docs_ref, pyclass));
    anstream::eprintln!("{}", "✅ All files generated!".green());
}

fn handle_class(docs_ref: &Path, pyclass: &PyClass) {
    let name = &pyclass.python_name;
    let path = docs_ref.join(format!("{}.md", name.to_lowercase()));
    let full_path = format!("{}.{}", pyclass.module.as_ref().unwrap(), name);
    let new_content = get_new_content(name, &full_path);
    WriteKind::new(&path, &new_content).maybe_write(&path, new_content);
}
enum WriteKind {
    New,
    Unchanged,
    Updated,
}
impl WriteKind {
    fn new(path: &Path, new_content: &str) -> Self {
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
    fn maybe_write(self, path: &Path, new_content: String) {
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
fn get_new_content(name: &str, full_path: &str) -> String {
    format!(
        "# {name}

::: {full_path}
"
    )
}
