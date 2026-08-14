use std::path::{Components, Display, Path, PathBuf};
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

pub(super) struct Root(PathBuf);
impl Root {
    pub(super) fn new(path: PathBuf) -> Self {
        Self(path)
    }
    pub(super) fn display(&self) -> Display<'_> {
        self.0.display()
    }
    pub(super) fn join(&self, path: &str) -> PathBuf {
        self.0.join(path)
    }

    pub(super) fn iter_on_extension(&self, extension: &str) -> impl Iterator<Item = PathBuf> {
        WalkDir::new(&self.0)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_type().is_file())
            .filter(|entry| {
                entry.path().extension().and_then(|value| value.to_str()) == Some(extension)
            })
            .map(|entry| entry.into_path())
    }

    pub(super) fn make_relative<'a>(&self, path: &'a Path) -> Relative<'a> {
        Relative(path.strip_prefix(&self.0).unwrap())
    }
}

/// Only relative path can be normalized
pub(super) struct Relative<'a>(&'a Path);
impl<'a> Relative<'a> {
    pub(super) fn normalize(self) -> Normalized {
        fn normalize_os_str(os_str: &std::ffi::OsStr) -> String {
            os_str.to_str().unwrap().trim_start_matches('_').to_string()
        }
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
