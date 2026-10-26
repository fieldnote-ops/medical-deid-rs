use anyhow::{Context, Result, ensure};
use std::{
    fs,
    path::{Component, Path, PathBuf},
};

pub fn work_dir() -> Result<PathBuf> {
    use std::os::unix::fs::MetadataExt;
    let root = PathBuf::from(
        std::env::var("DEID_STORAGE_ROOT")
            .context("source scripts/env.sh before running the executable")?,
    );
    ensure!(root.is_absolute(), "storage root must be absolute");
    let root = root.canonicalize().context("storage root is unavailable")?;
    let mode = std::env::var("DEID_REQUIRE_EXTERNAL").unwrap_or_else(|_| "1".into());
    ensure!(
        mode == "0" || mode == "1",
        "DEID_REQUIRE_EXTERNAL must be 0 or 1"
    );
    if mode == "1" {
        let parent = root
            .parent()
            .context("storage root must be a mounted volume")?;
        ensure!(
            fs::metadata(&root)?.dev() != fs::metadata(parent)?.dev()
                && fs::metadata(&root)?.dev() != fs::metadata("/")?.dev(),
            "configured external storage volume is not mounted"
        );
    }
    let work = root.join("medical-deid-rs");
    ensure!(
        work.is_dir(),
        "configured workspace is unavailable; source scripts/env.sh"
    );
    let canonical = work.canonicalize()?;
    ensure!(
        canonical.starts_with(&root),
        "workspace symlink escapes configured storage"
    );
    Ok(canonical)
}

pub fn validate_output(path: &Path) -> Result<()> {
    validate_under(path, &work_dir()?)
}

fn validate_under(path: &Path, work: &Path) -> Result<()> {
    ensure!(
        path.is_absolute() && !path.components().any(|p| matches!(p, Component::ParentDir)),
        "output must be absolute without parent traversal"
    );
    let mut existing = path;
    while !existing.exists() {
        existing = existing.parent().context("invalid output path")?;
    }
    ensure!(
        existing.canonicalize()?.starts_with(work.canonicalize()?),
        "output must remain inside configured DEID_WORK"
    );
    Ok(())
}

pub fn configure_runtime() -> Result<()> {
    use cubecl_runtime::config::{CubeClRuntimeConfig, RuntimeConfig, cache::CacheConfig};
    let cache = work_dir()?.join("cache/cubecl");
    fs::create_dir_all(&cache)?;
    let mut config = CubeClRuntimeConfig::default();
    config.environment.path = CacheConfig::Directory(cache.clone());
    if !CubeClRuntimeConfig::try_set(config) {
        ensure!(
            CubeClRuntimeConfig::get().environment.path.root() == cache,
            "runtime cache differs from configured storage"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_parent_traversal_relative_paths_and_symlink_escape() {
        let base = std::env::temp_dir().join(format!("deid-path-test-{}", std::process::id()));
        fs::create_dir_all(base.join("work")).unwrap();
        fs::create_dir_all(base.join("outside")).unwrap();
        let work = base.join("work");
        assert!(validate_under(&work.join("new/report.json"), &work).is_ok());
        assert!(validate_under(Path::new("relative"), &work).is_err());
        assert!(validate_under(&work.join("../outside/file"), &work).is_err());
        let link = work.join("escape");
        std::os::unix::fs::symlink(base.join("outside"), &link).unwrap();
        assert!(validate_under(&link.join("file"), &work).is_err());
        fs::remove_dir_all(base).unwrap();
    }
}
