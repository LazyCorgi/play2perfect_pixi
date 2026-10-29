"""Scale URDF geometry and constant-density inertial parameters."""
import os
from pathlib import Path
import xml.etree.ElementTree as ET

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
    tree.getroot().set("name", tree.getroot().get("name") + "_scaled")
    ET.indent(tree, space="  ")
    return tree
