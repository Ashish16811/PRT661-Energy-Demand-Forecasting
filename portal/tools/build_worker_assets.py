from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import joblib

EXPECTED = {
    "NSW1": "RandomForest_A",
    "QLD1": "RandomForest_A",
    "VIC1": "RandomForest_A",
    "SA1": "RandomForest_B",
    "TAS1": "RandomForest_A",
    "WA": "RandomForest_B",
}


def _sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _resolve_pointer(models_root: Path, pointer: Path) -> tuple[Path, str]:
    raw = pointer.read_text(encoding="utf-8").strip().replace("\\", "/")
    if raw.startswith("/app/models/"):
        rel = raw[len("/app/models/"):]
    elif raw.startswith("models/"):
        rel = raw[len("models/"):]
    elif raw.endswith("/bundle.joblib"):
        rel = raw
    else:
        # Legacy pointer containing only a version directory.
        region, model = pointer.parts[-3], pointer.parts[-2]
        rel = f"{region}/{model}/{raw}/bundle.joblib"
    path = models_root.joinpath(*rel.split("/"))
    if not path.exists():
        raise FileNotFoundError(f"Pointer {pointer} resolves to missing bundle: {path}")
    return path, rel


def _local_stage2_root(stored, stage2_dir: Path) -> Path:
    p = Path(str(stored))
    if p.exists():
        return p.resolve()
    parts = str(stored).replace("\\", "/").split("/")
    if "stage2" in parts:
        idx = len(parts) - 1 - parts[::-1].index("stage2")
        cand = stage2_dir.joinpath(*parts[idx + 1:])
        if cand.exists():
            return cand.resolve()
    raise FileNotFoundError(f"Cannot resolve bundle Stage 2 root {stored!r} beneath {stage2_dir}")


def _add_bytes(tf: tarfile.TarFile, arcname: str, data: bytes) -> None:
    info = tarfile.TarInfo(arcname)
    info.size = len(data)
    info.mtime = int(datetime.now(timezone.utc).timestamp())
    tf.addfile(info, io.BytesIO(data))


def _add_tree(tf: tarfile.TarFile, source: Path, arcname: str) -> None:
    for path in sorted(source.rglob("*")):
        if path.is_dir():
            continue
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        rel = path.relative_to(source).as_posix()
        tf.add(path, arcname=f"{arcname}/{rel}", recursive=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the one-time Railway worker volume asset archive")
    ap.add_argument("--project-root", default=".", help="Final local portal project root")
    ap.add_argument("--output", default="PRT661_WORKER_ASSETS.tar.gz")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    models_root = root / "models"
    stage2_dir = root / "data" / "stage2"
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output

    if not models_root.exists():
        raise FileNotFoundError(models_root)
    if not stage2_dir.exists():
        raise FileNotFoundError(stage2_dir)

    model_items = []
    stage2_roots: dict[str, Path] = {}

    print("Validating the six approved production bundles...")
    for region, model in EXPECTED.items():
        pointer = models_root / region / model / "CURRENT.txt"
        if not pointer.exists():
            raise FileNotFoundError(pointer)
        bundle_path, rel = _resolve_pointer(models_root, pointer)
        print(f"  {region}/{model}: {bundle_path.name} ({bundle_path.stat().st_size / 1024**2:.2f} MB)")

        # Read only once locally so the deployment manifest carries provenance and
        # the archive fails early if the bundle cannot be loaded under the approved runtime.
        bundle = joblib.load(bundle_path)
        if not isinstance(bundle, dict) or not hasattr(bundle.get("estimator"), "predict"):
            raise RuntimeError(f"Invalid portal bundle: {bundle_path}")
        if bundle.get("model_name") not in (None, model):
            raise RuntimeError(f"Bundle model mismatch for {region}: {bundle.get('model_name')} != {model}")

        s2 = _local_stage2_root(bundle.get("stage2_root"), stage2_dir)
        data_dir = s2 / "data"
        required = [data_dir / "future_feature_frame_all_regions.csv"] + [
            data_dir / f"modelling_{r}_30min.csv" for r in EXPECTED
        ]
        missing = [p for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Stage 2 root {s2} is missing required runtime files: {missing[:3]}")

        try:
            stage2_rel = s2.relative_to(stage2_dir).as_posix()
        except ValueError:
            raise RuntimeError(f"Stage 2 root must be beneath {stage2_dir}: {s2}")
        stage2_roots[stage2_rel] = s2

        model_items.append({
            "region": region,
            "model_name": model,
            "model_version": str(bundle.get("model_version") or bundle_path.parent.name),
            "feature_set": str(bundle.get("feature_set") or ("B" if model.endswith("_B") else "A")),
            "training_start": str(bundle.get("training_start") or ""),
            "training_end": str(bundle.get("training_end") or ""),
            "stage2_root": f"/app/storage/data/stage2/{stage2_rel}".rstrip("/"),
            "bundle_relative_path": rel,
            "bundle_size_bytes": bundle_path.stat().st_size,
            "bundle_sha256": _sha256(bundle_path),
        })
        del bundle

    manifest = {
        "asset_format": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "production_mapping": EXPECTED,
        "models": model_items,
        "stage2_roots": sorted(stage2_roots),
        "notes": "Contains only the six approved production bundles plus Stage 2 runtime inputs; no raw archive or SQLite database.",
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    print(f"Building {output} ...")
    with tarfile.open(output, "w:gz", compresslevel=1) as tf:
        # Store portable CURRENT pointers instead of machine/Docker-specific paths.
        for item in model_items:
            region, model = item["region"], item["model_name"]
            rel = item["bundle_relative_path"]
            bundle_path = models_root.joinpath(*rel.split("/"))
            tf.add(bundle_path, arcname=f"models/{rel}", recursive=False)
            _add_bytes(tf, f"models/{region}/{model}/CURRENT.txt", rel.encode("utf-8"))

        for rel, s2 in stage2_roots.items():
            arc = "data/stage2" if rel in {"", "."} else f"data/stage2/{rel}"
            print(f"  adding Stage 2 runtime root: {s2} -> {arc}")
            _add_tree(tf, s2, arc)

        # A portable Stage 2 pointer is useful for diagnostics even though the
        # production inference code can rebase the root from each bundle.
        first_rel = sorted(stage2_roots)[0]
        _add_bytes(tf, "data/stage2/CURRENT_STAGE2.txt", first_rel.encode("utf-8"))
        _add_bytes(tf, "worker_asset_manifest.json", json.dumps(manifest, indent=2).encode("utf-8"))

    print("\nWorker asset archive created successfully")
    print(f"  Path: {output}")
    print(f"  Size: {output.stat().st_size / 1024**2:.2f} MB")
    print(f"  SHA256: {_sha256(output)}")
    print("  Contains: 6 production bundles + required Stage 2 runtime snapshot")
    print("  Excludes: data/raw, data/runtime/portal.db, research/non-production models")


if __name__ == "__main__":
    main()
