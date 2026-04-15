from __future__ import annotations

import os
import shutil
from pathlib import Path

import cocopp


REPO_ROOT = Path(__file__).resolve().parent
BUILD_ROOT = REPO_ROOT / ".coco_external_build"
INPUT_ROOT = BUILD_ROOT
PPDATA_ROOT = REPO_ROOT / "ppdata"
TARGET_DIR = PPDATA_ROOT / "external_refs_data"
TARGET_LINK = PPDATA_ROOT / "external_refs"

ARCHIVES = {
    "cmaes.tgz": "2020/CMA-ES-2019_Hansen.tgz",
    "lqcmaes.tgz": "2020/lq-CMA-ES_Hansen.tgz",
    "dtscmaes.tgz": "2018/DTS-CMA-ES_005-2pop_v26_1model_Bajer.tgz",
}

LOCAL_INPUTS = {
    "doevae": REPO_ROOT / "exdata" / "doe28",
    "nn": REPO_ROOT / "exdata" / "nn3",
}


def recreate_symlink(link_path: Path, target: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    link_path.symlink_to(target)


def prepare_inputs() -> list[str]:
    if BUILD_ROOT.exists():
        shutil.rmtree(BUILD_ROOT)
    INPUT_ROOT.mkdir(parents=True)

    for alias, local_path in LOCAL_INPUTS.items():
        recreate_symlink(INPUT_ROOT / alias, local_path)

    for alias, archive_id in ARCHIVES.items():
        archive_path = Path(cocopp.bbob.get(archive_id))
        recreate_symlink(INPUT_ROOT / alias, archive_path)

    return [
        "doevae",
        "nn",
        "cmaes.tgz",
        "lqcmaes.tgz",
        "dtscmaes.tgz",
    ]


def find_generated_folder() -> Path:
    ppdata_dir = BUILD_ROOT / "ppdata"
    candidates = [path for path in ppdata_dir.iterdir() if path.is_dir()]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one generated ppdata folder, got {candidates!r}")
    return candidates[0]


def install_bundle(generated_dir: Path) -> None:
    PPDATA_ROOT.mkdir(exist_ok=True)
    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)
    shutil.move(str(generated_dir), str(TARGET_DIR))
    recreate_symlink(TARGET_LINK, TARGET_DIR.name)


def main() -> None:
    argv = prepare_inputs()
    cocopp.genericsettings.isExpensive = True
    cocopp.genericsettings.xlimit_expensive = 250.0
    cocopp.genericsettings.isConv = True

    cwd = Path.cwd()
    try:
        os.chdir(BUILD_ROOT)
        cocopp.main(" ".join(argv))
    finally:
        os.chdir(cwd)

    install_bundle(find_generated_folder())
    print(TARGET_DIR)


if __name__ == "__main__":
    main()