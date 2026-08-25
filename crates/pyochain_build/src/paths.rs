use std::{
    fs,
    path::{Components, Display, Path, PathBuf},
};
use tap::Pipe;
use walkdir::WalkDir;
#[derive(Clone, Debug, Eq)]
pub(super) struct Normalized {
    source: PathBuf,
    parent: Vec<String>,
    stem: String,
}
impl PartialEq for Normalized {
    fn eq(&self, other: &Self) -> bool {
        self.parent == other.parent && self.stem == other.stem
    }
}
impl Normalized {
    pub(super) fn display(&self) -> Display<'_> {
        self.source.display()
    }
}

pub(super) struct Related {
    pub parent: PathBuf,
    pub child: PathBuf,
    extension: String,
}
impl Related {
    pub(super) fn new(parent: PathBuf, child: &str, extension: &str) -> Self {
        let child = parent.join(child);
        Self {
            parent,
            child,
            extension: extension.to_string(),
        }
    }
    pub(super) fn display(&self) -> Display<'_> {
        self.parent.display()
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = PathBuf> {
        WalkDir::new(&self.parent)
            .into_iter()
            .map(|file| file.expect("Failed to read file entry"))
            .filter(|entry| {
                entry.file_type().is_file()
                    && entry.path().extension().and_then(|value| value.to_str())
                        == Some(self.extension.as_str())
            })
            .map(walkdir::DirEntry::into_path)
    }

    pub(super) fn make_relative<'a>(&self, path: &'a Path) -> Relative<'a> {
        Relative(path.strip_prefix(&self.parent).unwrap())
    }

    pub fn write(&self, path: PathBuf, content: String) -> Result<String, std::io::Error> {
        let display_path = self.make_relative(&path).0.display();
        let result = match path.exists() {
            false => {
                fs::write(&path, content)?;
                format!("Generated {display_path} (new file)")
            }
            true if fs::read_to_string(&path)? == content => {
                format!("Skipping {display_path} (no changes)")
            }
            true => {
                fs::write(&path, content)?;
                format!("Updating {display_path} (co§ntent changed)")
            }
        };
        Ok(result)
    }
}

/// Only relative path can be normalized
pub(super) struct Relative<'a>(pub &'a Path);
impl Relative<'_> {
    pub(super) fn normalize(self) -> Normalized {
        Normalized {
            source: self.0.to_path_buf(),
            parent: self
                .components()
                .map(|component| component.as_os_str().pipe(normalize_os_str))
                .collect(),
            stem: self.0.file_stem().unwrap().pipe(normalize_os_str),
        }
    }
    pub(super) fn components(&self) -> Components<'_> {
        self.0.parent().unwrap().components()
    }
}

fn normalize_os_str(os_str: &std::ffi::OsStr) -> String {
    os_str.to_str().unwrap().trim_start_matches('_').to_string()
}
