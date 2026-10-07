"""Apply the Level 3 CUDA/headless fix without Git or extra packages.

Run from the project root:
    python scripts/apply_level3_headless_cuda.py --check
    python scripts/apply_level3_headless_cuda.py

All edits are checked before any file is changed. Original files are backed up
under records/level3_patch_backups. Already-applied edits are accepted.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import tempfile


CHANGES = {
    "sim/generate/builder.py": [(
        "\n        viewer_options=us.options.ViewerOptions(),\n    )\n",
        "\n        viewer_options=us.options.ViewerOptions(),\n"
        "        enable_visualization=config.render,\n    )\n",
    )],
    "sim/mpm/__init__.py": [(
        "\n    ti.init(arch=ti.gpu, device_memory_GB=allocate_gpu_memory, debug=debug)\n",
        "\n    ti.init(arch=ti.cuda, device_memory_GB=allocate_gpu_memory,\n"
        "            debug=debug, enable_fallback=False)\n",
    )],
    "sim/mpm/engine/scene.py": [(
        "\n            renderer_options = RendererOptions(),\n        ):\n",
        "\n            renderer_options = RendererOptions(),\n"
        "            enable_visualization = True,\n        ):\n",
    ), (
        "\n        self.visualizer = Visualizer(\n"
        "            viewer_options   = viewer_options,\n"
        "            renderer_options = renderer_options,\n        )\n",
        "\n        self.visualizer = None\n        if enable_visualization:\n"
        "            self.visualizer = Visualizer(\n"
        "                viewer_options   = viewer_options,\n"
        "                renderer_options = renderer_options,\n            )\n",
    ), (
        "\n        return self.visualizer.add_camera(res, pos, lookat, up, fov)\n",
        "\n        if self.visualizer is None:\n"
        "            raise RuntimeError('Cameras require enable_visualization=True.')\n"
        "        return self.visualizer.add_camera(res, pos, lookat, up, fov)\n",
    ), (
        "\n        self.visualizer.build(self)\n",
        "\n        if self.visualizer is not None:\n            self.visualizer.build(self)\n",
    ), (
        "\n    def get_viewer_image(self):\n        return self.visualizer.viewer.get_image()\n",
        "\n    def get_viewer_image(self):\n        if self.visualizer is None:\n"
        "            raise RuntimeError('Viewer images require enable_visualization=True.')\n"
        "        return self.visualizer.viewer.get_image()\n",
    )],
}


def plan_changes(root):
    plan = []
    for relative, replacements in CHANGES.items():
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError(f"Target is outside the project: {relative}")
        original = path.read_bytes()
        updated = original.decode("utf-8-sig").replace("\r\n", "\n")
        count = 0
        for index, (old, new) in enumerate(replacements, 1):
            old_count, new_count = updated.count(old), updated.count(new)
            if new_count == 1 and updated.replace(new, "", 1).count(old) == 0:
                continue
            elif old_count == 1 and new_count == 0:
                updated = updated.replace(old, new, 1)
                count += 1
            else:
                raise ValueError(
                    f"Content does not match the expected version: {relative}, edit {index}. "
                    "No source files have been changed."
                )
        compile(updated, str(path), "exec")
        print(f"{relative}: {count} pending edits" if count else f"{relative}: already applied")
        if count:
            if b"\r\n" in original:
                updated = updated.replace("\n", "\r\n")
            encoding = "utf-8-sig" if original.startswith(b"\xef\xbb\xbf") else "utf-8"
            plan.append((relative, path, original, updated.encode(encoding)))
    return plan


def apply_changes(root, plan):
    backup_parent = root / "records/level3_patch_backups"
    backup_parent.mkdir(parents=True, exist_ok=True)
    backup_dir = Path(tempfile.mkdtemp(prefix="headless_cuda_", dir=str(backup_parent)))
    for relative, path, original, _ in plan:
        if path.read_bytes() != original:
            raise RuntimeError(f"File changed during checking: {relative}. Retry the check.")
        backup = backup_dir / relative
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
        if backup.read_bytes() != original:
            raise RuntimeError(f"Backup verification failed: {relative}")
    print(f"Original files backed up to: {backup_dir}")
    changed = []
    try:
        for relative, path, original, updated in plan:
            if path.read_bytes() != original:
                raise RuntimeError(f"File changed before applying the fix: {relative}")
            with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(updated)
            try:
                shutil.copystat(path, temporary)
                os.replace(temporary, path)
                changed.append((relative, path))
            finally:
                temporary.unlink(missing_ok=True)
    except Exception:
        for relative, path in reversed(changed):
            shutil.copy2(backup_dir / relative, path)
        raise
    print("PATCH_APPLIED")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check only; do not create backups or edit files.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    root = args.project_root.resolve()
    try:
        plan = plan_changes(root)
        if args.check:
            print("CHECK_PASS")
        elif not plan:
            print("PATCH_ALREADY_APPLIED")
        else:
            apply_changes(root, plan)
    except (OSError, UnicodeError, ValueError, SyntaxError, RuntimeError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
