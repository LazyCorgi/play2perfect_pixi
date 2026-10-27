"""Independent rigid-box audit for insertion, separate from reward keypoints.

All poses are in the receiver frame, xyzw. This checks the shipped primitive
geometry, not elastic deformation, manufacturing error or measured force.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def collision_boxes(path: Path):
    boxes = []
    root = ET.parse(path).getroot()
    if root.findall("joint"):
        raise ValueError("This audit supports only the single-link peg/receiver")
    for element in root.findall("./link/collision"):
        box = element.find("geometry/box")
        if box is None:
            raise ValueError("Insertion audit requires box colliders")
        origin = element.find("origin")
        xyz = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
        rpy = np.fromstring(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ")
        boxes.append((xyz, Rotation.from_euler("xyz", rpy).as_matrix(), np.fromstring(box.get("size"), sep=" ") / 2))
    if not boxes:
        raise ValueError("No collision boxes")
    return boxes


def box_overlap(a, b) -> float:
    """Separating-axis overlap in meters; zero for separated or touching boxes."""
    ca, ra, ha = a
    cb, rb, hb = b
    axes = np.concatenate((ra.T, rb.T, np.cross(ra.T[:, None], rb.T[None, :]).reshape(-1, 3)))
    norms = np.linalg.norm(axes, axis=1)
    axes = axes[norms > 1e-10] / norms[norms > 1e-10, None]
    overlap = np.abs(axes @ ra) @ ha + np.abs(axes @ rb) @ hb - np.abs(axes @ (cb - ca))
    return max(0.0, float(overlap.min()))


class PegInsertionGeometry:
    def __init__(self, object_urdf: Path, receiver_urdf: Path, final_pose):
        self.object_boxes = collision_boxes(object_urdf)
        self.fixture_boxes = collision_boxes(receiver_urdf)
        self.final_pose = np.asarray(final_pose, dtype=float)
        self.final_rotation = Rotation.from_quat(self.final_pose[3:])
        self.corners = np.asarray(list(product((-1, 1), repeat=3)))
        self.expected_tip_z = self._tip_z(self.final_pose)

    def _tip_z(self, pose):
        # The first collider is the main rectangular shaft in the shipped peg.
        center, local_rotation, half = self.object_boxes[0]
        corners = center + (self.corners * half) @ local_rotation.T
        return float((corners @ Rotation.from_quat(pose[3:]).as_matrix().T + pose[:3])[:, 2].min())

    def evaluate(self, pose, *, overlap_tolerance_m=0.00005, seating_tolerance_m=0.001, angle_tolerance_deg=2.0):
        pose = np.asarray(pose, dtype=float)
        rotation = Rotation.from_quat(pose[3:])
        matrix = rotation.as_matrix()
        transformed = [(pose[:3] + matrix @ center, matrix @ axes, half) for center, axes, half in self.object_boxes]
        overlap = max(box_overlap(a, b) for a in transformed for b in self.fixture_boxes)
        tip_error = abs(self._tip_z(pose) - self.expected_tip_z)
        position_error = float(np.linalg.norm(pose[:3] - self.final_pose[:3]))
        angle_error = float(np.degrees((self.final_rotation.inv() * rotation).magnitude()))
        seated = (overlap <= overlap_tolerance_m and tip_error <= seating_tolerance_m
                  and position_error <= seating_tolerance_m and angle_error <= angle_tolerance_deg)
        return {"seated": bool(seated), "box_overlap_m": overlap, "tip_height_error_m": tip_error,
                "position_error_m": position_error, "angle_error_deg": angle_error}
