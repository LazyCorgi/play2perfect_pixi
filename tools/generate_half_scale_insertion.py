"""Generate the half-size peg and receiver with constant-density inertial scaling.

Run from any directory with the project's Python. Original assets are never edited.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets/urdf/peg_in_hole"
OUTPUT = ASSETS / "half_scale"
FACTOR = 0.5


def scaled_urdf(source: Path, factor: float, *, destination: Path | None = None, mass_factor: float | None = None) -> ET.ElementTree:
    if factor <= 0:
        raise ValueError("Scale must be positive")
    tree = ET.parse(source)
    mass_factor = factor**3 if mass_factor is None else mass_factor
    if mass_factor <= 0:
        raise ValueError("Mass factor must be positive")
    for element in tree.iter():
        # Both collision/visual origins and the COM use length coordinates.
        attributes = {
            "origin": ("xyz",), "box": ("size",),
            "cylinder": ("radius", "length"), "sphere": ("radius",),
        }.get(element.tag, ())
        for name in attributes:
            if name in element.attrib:
                element.set(name, " ".join(f"{float(v) * factor:.12g}" for v in element.get(name).split()))
        if element.tag == "mesh":
            if destination is None:
                raise ValueError("Mesh scaling requires a destination for rewriting relative paths")
            mesh = (source.parent / element.get("filename")).resolve()
            if not mesh.is_file():
                raise FileNotFoundError(mesh)
            element.set("filename", os.path.relpath(mesh, destination.parent))
            scale = [float(v) for v in element.get("scale", "1 1 1").split()]
            element.set("scale", " ".join(f"{v * factor:.12g}" for v in scale))
        if element.tag == "mass":
            element.set("value", f"{float(element.get('value')) * mass_factor:.12g}")
        if element.tag == "inertia":
            for name, value in element.attrib.items():
                element.set(name, f"{float(value) * mass_factor * factor**2:.12g}")
    tree.getroot().set("name", tree.getroot().get("name") + "_half")
    ET.indent(tree, space="  ")
    return tree


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pairs = (
        (ASSETS / "lpeg_matchedmass/lpeg_matchedmass.urdf", OUTPUT / "lpeg_matchedmass_half.urdf"),
        (ASSETS / "holes/hole_tol0p5mm/hole_tol0p5mm.urdf", OUTPUT / "hole_tol0p25mm.urdf"),
    )
    sources = {}
    for source, target in pairs:
        scaled_urdf(source, FACTOR).write(target, encoding="utf-8", xml_declaration=True)
        sources[str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
        print(target.relative_to(ROOT))
    # Controlled dynamics ablation: half geometry with the ORIGINAL mass.
    # I scales by m_ratio * length_ratio**2, so it must be 1/4, not 1/32.
    heavy = OUTPUT / "lpeg_matchedmass_half_original_mass.urdf"
    scaled_urdf(pairs[0][0], FACTOR, mass_factor=1.0).write(heavy, encoding="utf-8", xml_declaration=True)
    intermediate = ASSETS / "scale_075"
    intermediate.mkdir(exist_ok=True)
    scaled_urdf(pairs[0][0], .75).write(intermediate / "lpeg_075.urdf", encoding="utf-8", xml_declaration=True)
    scaled_urdf(pairs[1][0], .75).write(intermediate / "hole_075.urdf", encoding="utf-8", xml_declaration=True)
    metadata = {
        "problem": "tight_insertion_half", "source_problem": "tight_insertion",
        "linear_scale": FACTOR, "mass_scale": FACTOR**3, "inertia_scale": FACTOR**5,
        "mass_assumption": "constant density; validate against manufactured parts before deployment",
        "nominal_side_clearance_m": 0.00025,
        "grasp_box_policy_scale": [3.125, 0.375, 0.25],
        "source_sha256": sources,
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
