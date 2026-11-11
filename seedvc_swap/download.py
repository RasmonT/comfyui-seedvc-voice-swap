"""Locate, download (with resume) and verify model files. Standard library only."""
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from .manifest import FILES, ModelFile, required_keys, total_size

CHUNK = 4 * 1024 * 1024
USER_AGENT = "comfyui-seedvc-voice-swap"


class ModelError(RuntimeError):
    """A model file is missing, corrupt or could not be downloaded."""


def sha256_file(path, progress=None):
    digest = hashlib.sha256()
    done = 0
    with open(path, "rb") as fh:
        while True:
            block = fh.read(CHUNK)
            if not block:
                break
            digest.update(block)
            done += len(block)
            if progress:
                progress(done)
    return digest.hexdigest()


class _VerifiedCache:
    """Remembers which (path, size, mtime) already passed the SHA256 check."""

    def __init__(self, cache_file):
        self.file = Path(cache_file)
        try:
            self.data = json.loads(self.file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.data = {}

    @staticmethod
    def _stamp(path):
        st = os.stat(path)
        return [st.st_size, st.st_mtime_ns]

    def is_verified(self, path, sha256):
        entry = self.data.get(str(path))
        return bool(entry) and entry.get("sha256") == sha256 and entry.get("stamp") == self._stamp(path)

    def mark(self, path, sha256):
        self.data[str(path)] = {"sha256": sha256, "stamp": self._stamp(path)}
        try:
            self.file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.file.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
            os.replace(tmp, self.file)
        except OSError:
            pass  # the cache only saves time; failing to write it is harmless


def _verify(path, spec: ModelFile, cache, log):
    size = os.path.getsize(path)
    if size != spec.size:
        raise ModelError(
            f"Model file has the wrong size: {path}\n  expected {spec.size} bytes, found {size} bytes.\n"
            f"  Delete the file and run again to download it afresh.")
    if cache.is_verified(path, spec.sha256):
        return
    log(f"[SeedVC] verifying SHA256 of {Path(path).name} ...")
    actual = sha256_file(path)
    if actual != spec.sha256:
        raise ModelError(
            f"Model file failed the SHA256 check: {path}\n  expected {spec.sha256}\n  found    {actual}\n"
            f"  The file is corrupt or is not the official one. Delete it and run again.")
    cache.mark(path, spec.sha256)


def _download(spec: ModelFile, dest: Path, log, progress=None, retries=5):
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last_error = None
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        if have > spec.size:
            part.unlink()
            have = 0
        if have == spec.size:
            break
        request = urllib.request.Request(spec.url, headers={"User-Agent": USER_AGENT})
        if have:
            request.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                resumed = have and response.status == 206
                if have and not resumed:
                    have = 0  # server ignored the Range header; start over
                log(f"[SeedVC] downloading {dest.name} ({spec.size / 1e6:.1f} MB)"
                    + (f", resuming at {have / 1e6:.1f} MB" if resumed else ""))
                with open(part, "ab" if resumed else "wb") as fh:
                    while True:
                        block = response.read(CHUNK)
                        if not block:
                            break
                        fh.write(block)
                        have += len(block)
                        if progress:
                            progress(len(block))
            if have == spec.size:
                break
            last_error = f"connection closed after {have} of {spec.size} bytes"
        except urllib.error.HTTPError as exc:
            if exc.code == 416:  # range not satisfiable: the partial file is unusable
                part.unlink(missing_ok=True)
            last_error = f"HTTP {exc.code} {exc.reason}"
            if exc.code in (401, 403, 404):
                break
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_error = str(exc)
        time.sleep(min(2 ** attempt, 20))
    else:
        raise ModelError(
            f"Could not download {dest.name} after {retries} attempts ({last_error}).\n  URL: {spec.url}\n"
            f"  The partial file was kept; run again to resume.")
    if not part.exists() or part.stat().st_size != spec.size:
        raise ModelError(f"Could not download {dest.name} ({last_error}).\n  URL: {spec.url}")
    actual = sha256_file(part)
    if actual != spec.sha256:
        part.unlink(missing_ok=True)
        raise ModelError(
            f"Downloaded file failed the SHA256 check and was deleted: {dest.name}\n"
            f"  expected {spec.sha256}\n  found    {actual}\n  URL: {spec.url}")
    os.replace(part, dest)


def ensure_models(models_dir, mode, keep_background, wan2gp_ckpts="", log=print, progress=None,
                  allow_download=True):
    """Return {key: absolute path} for every file the requested run needs.

    With `wan2gp_ckpts` pointing at an existing folder, files are taken from it and nothing is
    downloaded; a file missing there or failing its SHA256 is an error. A path that does not
    exist is ignored with a warning. Otherwise missing files are downloaded into `models_dir`
    (resumable) and verified.
    """
    keys = required_keys(mode, keep_background)
    models_dir = Path(models_dir)
    cache = _VerifiedCache(models_dir / "verified.json")
    resolved = {}

    wan2gp_ckpts = (wan2gp_ckpts or "").strip().strip('"')
    if wan2gp_ckpts and not Path(wan2gp_ckpts).is_dir():
        # The path is saved inside workflows, so a shared workflow can carry somebody else's folder.
        # A folder that is not there is treated like an empty field instead of stopping the run.
        log(f"[SeedVC] WARNING: wan2gp_ckpts_path does not exist (or is not a folder) here: {wan2gp_ckpts}\n"
            f"[SeedVC]          Ignoring it and using {models_dir} (missing models are downloaded there). "
            f"Clear the field to silence this warning.")
        wan2gp_ckpts = ""
    if wan2gp_ckpts:
        root = Path(wan2gp_ckpts)
        missing = [FILES[k].wan2gp for k in keys if not (root / FILES[k].wan2gp).is_file()]
        if missing:
            raise ModelError(
                f"The Wan2GP ckpts folder is missing {len(missing)} required file(s):\n  "
                + "\n  ".join(missing)
                + f"\n  Folder: {root}\n  Run this mode once in Wan2GP so it downloads them, "
                  f"or clear the field so this node downloads its own copies.")
        for k in keys:
            path = root / FILES[k].wan2gp
            _verify(path, FILES[k], cache, log)
            resolved[k] = str(path)
        return resolved

    missing = [k for k in keys if not (models_dir / FILES[k].path).is_file()]
    if missing and not allow_download:
        raise ModelError(
            "Model files are missing and downloading is disabled:\n  "
            + "\n  ".join(str(models_dir / FILES[k].path) for k in missing))
    if missing:
        log(f"[SeedVC] {len(missing)} model file(s) missing, downloading {total_size(missing) / 1e6:.0f} MB "
            f"into {models_dir}")
    for k in keys:
        spec = FILES[k]
        path = models_dir / spec.path
        if not path.is_file():
            _download(spec, path, log, progress)
            cache.mark(path, spec.sha256)
        else:
            _verify(path, spec, cache, log)
        resolved[k] = str(path)
    return resolved
