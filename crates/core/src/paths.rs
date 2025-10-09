use std::env;
use std::path::{Path, PathBuf};

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
