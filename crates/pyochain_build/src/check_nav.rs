use crate::paths::Root;
use owo_colors::OwoColorize;
use regex::Regex;
use std::{collections::HashSet, fs, path::Path, sync::LazyLock};
use tap::Pipe;
use toml::{Table, Value};
use url::Url;

const DOCS_SITE_URL: &str = "https://outsquarecapital.github.io/pyochain/";

static FENCED_CODE_BLOCK: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)```.*?```").unwrap());
static MARKDOWN_LINK: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[[^\]]+\]\(([^)]+)\)").unwrap());

pub(super) fn run(root: &Root) {
    anstream::eprintln!("{}", "Checking navigation completeness...".cyan().bold());
    let project = get_project(root);
    let docs_dirs = project["docs_dir"]
        .as_str()
        .expect("Failed to get `docs_dir` from zensical.toml");
    check_all(&project, &root.join(docs_dirs), root);
}

fn check_all(project: &Table, docs_dir: &Path, root: &Root) {
    let mut nav_paths = HashSet::new();
    collect_nav_paths(&project["nav"], &mut nav_paths);
    let missing_paths = missing_paths(&docs_dir, &nav_paths);
    let invalid_nav_paths = invalid_nav_paths(&docs_dir, &nav_paths);
    let invalid_markdown_links = invalid_markdown_links(root, &docs_dir);

    if !missing_paths.is_empty() {
        show_warning(format_args!(
            "⚠️  Missing generated files in zensical.toml:\n {missing_paths}"
        ));
    }
    if !invalid_nav_paths.is_empty() {
        show_warning(format_args!("⚠️  Invalid nav links:\n {invalid_nav_paths}"));
    }
    if !invalid_markdown_links.is_empty() {
        show_warning(format_args!(
            "⚠️  Invalid markdown links:\n {invalid_markdown_links}"
        ));
    }

    if missing_paths.is_empty() && invalid_nav_paths.is_empty() && invalid_markdown_links.is_empty()
    {
        anstream::eprintln!("{}", "✓ Navigation is complete!".green());
    } else {
        anstream::eprintln!(
            "{}",
            "❌ Please fix the above issues before deploying the documentation."
                .red()
                .bold()
        );
    }
}
fn get_project(root: &Root) -> toml::value::Table {
    let project = root
        .join("zensical.toml")
        .pipe(fs::read_to_string)
        .expect("Failed to read zensical.toml")
        .parse::<toml::Table>()
        .expect("Failed to parse zensical.toml")
        .remove("project")
        .expect("Failed to get `project` key from parsed file");
    match project {
        Value::Table(table) => table,
        _ => panic!("Invalid zensical.toml: 'project' should be a table"),
    }
}
fn collect_nav_paths(value: &Value, paths: &mut HashSet<String>) {
    match value {
        Value::Table(table) => table
            .values()
            .for_each(|value| collect_nav_paths(value, paths)),
        Value::Array(values) => values
            .iter()
            .for_each(|value| collect_nav_paths(value, paths)),
        Value::String(path) => {
            paths.insert(path.clone());
        }
        _ => {}
    }
}

fn missing_paths(docs_dir: &Path, nav_paths: &HashSet<String>) -> String {
    docs_dir
        .join("reference")
        .pipe(fs::read_dir)
        .expect("Failed to read reference directory")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|extension| extension.to_str()) == Some("md"))
        .map(|path| relative_path(&path, docs_dir))
        .collect::<HashSet<_>>()
        .difference(nav_paths)
        .pipe(join_paths)
}

fn invalid_nav_paths(docs_dir: &Path, nav_paths: &HashSet<String>) -> String {
    let docs_paths = Root::new(docs_dir.to_path_buf())
        .iter_on_extension("md")
        .map(|path| relative_path(&path, docs_dir))
        .collect::<HashSet<_>>();
    nav_paths.difference(&docs_paths).pipe(join_paths)
}

fn invalid_markdown_links(root: &Root, docs_dir: &Path) -> String {
    [root.join("README.md"), root.join("CONTRIBUTING.md")]
        .into_iter()
        .flat_map(|doc_path| {
            let content = doc_path
                .pipe_ref(fs::read_to_string)
                .expect("Failed to read file");
            let content = FENCED_CODE_BLOCK.replace_all(&content, "");
            MARKDOWN_LINK
                .captures_iter(&content)
                .map(|captures| {
                    (
                        doc_path.clone(),
                        normalize_link_target(captures.get(1).unwrap().as_str()),
                    )
                })
                .collect::<Vec<_>>()
        })
        .filter(|(doc_path, target)| !is_valid_markdown_link(doc_path, target, docs_dir))
        .map(|(doc_path, target)| format!("{} -> {target}", doc_path.display()))
        .collect::<Vec<_>>()
        .join("\n")
}

fn is_valid_markdown_link(doc_path: &Path, target: &str, docs_dir: &Path) -> bool {
    let normalized_target = target.split('#').next().unwrap();
    match normalized_target {
        "" => true,
        target if target.starts_with(DOCS_SITE_URL) => is_valid_docs_site_link(target, docs_dir),
        target if target.starts_with("http://") || target.starts_with("https://") => true,
        target => doc_path.parent().unwrap().join(target).exists(),
    }
}

fn is_valid_docs_site_link(target: &str, docs_dir: &Path) -> bool {
    let parsed_target = Url::parse(target).expect("Failed to parse URL");
    let relative_path = parsed_target
        .path()
        .strip_prefix("/pyochain/")
        .unwrap_or(parsed_target.path())
        .trim_matches('/');
    match relative_path {
        "" => docs_dir.join("index.md").exists(),
        _ if parsed_target.path().ends_with('/') => {
            docs_dir.join(relative_path).with_extension("md").exists()
        }
        _ => docs_dir.join(relative_path).exists(),
    }
}

fn normalize_link_target(target: &str) -> String {
    let target = target.trim();
    let target = target.strip_prefix('<').unwrap_or(target);
    target.strip_suffix('>').unwrap_or(target).to_string()
}

fn relative_path(path: &Path, root: &Path) -> String {
    path.strip_prefix(root)
        .expect("Failed to strip root prefix from path")
        .to_string_lossy()
        .replace('\\', "/")
}

fn join_paths<'a>(paths: impl Iterator<Item = &'a String>) -> String {
    paths.map(String::as_str).collect::<Vec<_>>().join("\n")
}

fn show_warning(message: impl std::fmt::Display) {
    anstream::eprintln!("{}", message.yellow());
}
