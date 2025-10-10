use std::env;
use std::ffi::OsStr;
use std::fs;
use std::path::{Component, Path, PathBuf};

use serde::Deserialize;

fn env_project_root() -> Option<PathBuf> {
    for var in ["KLARNET_HOME", "KLARNET_ROOT"] {
        if let Ok(value) = env::var(var) {
            let candidate = PathBuf::from(value);
            if candidate.is_dir() {
                return Some(candidate);
            }
        }
    }
    None
}

fn is_project_root(dir: &Path) -> bool {
    dir.join(".venv").is_dir()
        || dir.join("config").join("klarnet.toml").is_file()
        || dir.join("Cargo.toml").is_file()
}

fn find_root_upwards(start: &Path) -> Option<PathBuf> {
    for ancestor in start.ancestors() {
        if is_project_root(ancestor) {
            return Some(ancestor.to_path_buf());
        }
    }
    None
}

/// Try to detect the project root directory using environment variables or the current executable.
pub fn detect_project_root() -> Option<PathBuf> {
    if let Some(root) = env_project_root() {
        return Some(root);
    }

    if let Ok(exe_path) = env::current_exe() {
        if let Some(dir) = exe_path.parent() {
            if let Some(root) = find_root_upwards(dir) {
                return Some(root);
            }
        }
    }

    if let Ok(cwd) = env::current_dir() {
        if let Some(root) = find_root_upwards(&cwd) {
            return Some(root);
        }
    }

    None
}

fn python_candidates_from_venv(venv_dir: &Path) -> [PathBuf; 5] {
    [
        venv_dir.join("Scripts").join("python.exe"),
        venv_dir.join("Scripts").join("python"),
        venv_dir.join("bin").join("python3"),
        venv_dir.join("bin").join("python"),
        venv_dir.join("python.exe"),
    ]
}

fn default_system_python() -> PathBuf {
    if cfg!(windows) {
        PathBuf::from("python.exe")
    } else {
        PathBuf::from("python3")
    }
}

fn detect_python_from_env() -> Option<PathBuf> {
    if let Ok(path) = env::var("KLARNET_PYTHON_PATH") {
        let candidate = PathBuf::from(&path);
        if candidate.exists() {
            return Some(candidate);
        }
    }

    if let Ok(venv) = env::var("VIRTUAL_ENV") {
        let venv_path = PathBuf::from(venv);
        for candidate in python_candidates_from_venv(&venv_path) {
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }

    None
}

/// Resolve the Python interpreter path that should be used by KLARNET.
///
/// The lookup order is:
/// 1. `KLARNET_PYTHON_PATH` environment variable (if it points to an existing file)
/// 2. Python inside the discovered project `.venv`
/// 3. `VIRTUAL_ENV` environment variable (if set)
/// 4. Platform default (`python.exe` on Windows, `python3` elsewhere)
pub fn resolve_python_path() -> PathBuf {
    if let Some(env_path) = detect_python_from_env() {
        return env_path;
    }

    if let Some(root) = detect_project_root() {
        let venv_dir = root.join(".venv");
        for candidate in python_candidates_from_venv(&venv_dir) {
            if candidate.exists() {
                return candidate;
            }
        }
    }

    default_system_python()
}

/// Resolve a relative path against the detected project root, if available.
///
/// If the provided path is already absolute or the project root cannot be detected,
/// the original path is returned unchanged.
pub fn resolve_project_path<P: AsRef<Path>>(path: P) -> PathBuf {
    let path = path.as_ref();
    if path.is_absolute() {
        return path.to_path_buf();
    }

    if let Some(root) = detect_project_root() {
        return root.join(path);
    }

    path.to_path_buf()
}

/// Resolve a path located inside the project's `scripts` directory.
///
/// Relative paths that already start with `scripts/` are resolved against the
/// project root unchanged. Bare filenames (e.g. `whisper_server.py`) are
/// considered to live directly inside the `scripts` folder. Absolute paths are
/// returned verbatim.
pub fn resolve_scripts_path<P: AsRef<Path>>(path: P) -> PathBuf {
    let path = path.as_ref();

    if path.is_absolute() {
        return path.to_path_buf();
    }

    let mut components = path.components();
    if let Some(Component::Normal(first)) = components.next() {
        if first == OsStr::new("scripts") {
            return resolve_project_path(path);
        }
    }

    resolve_project_path(Path::new("scripts").join(path))
}

/// Convenience accessor that resolves the project `scripts` directory itself.
pub fn resolve_scripts_dir() -> PathBuf {
    resolve_project_path("scripts")
}

/// Deserialize a project-relative path, resolving it against the detected
/// project root when necessary.
pub fn deserialize_project_path<'de, D>(deserializer: D) -> Result<PathBuf, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw = PathBuf::deserialize(deserializer)?;
    Ok(resolve_project_path(raw))
}

/// Return potential site-packages directories inside the project's virtual environment.
///
/// The lookup mirrors both Windows-style (`Lib/site-packages`) and Unix-style
/// (`lib/python*/site-packages`) layouts. Non-existent directories are skipped
/// silently and the resulting list may be empty when no virtual environment is
/// detected.
pub fn venv_site_packages_directories() -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    if let Some(root) = detect_project_root() {
        let venv_root = root.join(".venv");
        if !venv_root.is_dir() {
            return dirs;
        }

        let windows_layout = venv_root.join("Lib").join("site-packages");
        if windows_layout.is_dir() {
            dirs.push(windows_layout);
        }

        let unix_lib = venv_root.join("lib");
        if unix_lib.is_dir() {
            if let Ok(entries) = fs::read_dir(&unix_lib) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if !path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .map(|name| name.starts_with("python"))
                        .unwrap_or(false)
                    {
                        continue;
                    }

                    let site_packages = path.join("site-packages");
                    if site_packages.is_dir() {
                        dirs.push(site_packages);
                    }
                }
            }
        }
    }

    dirs
}