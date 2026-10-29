"""Names and registration for physical beam size variants (percent of release)."""

import dataclasses
import re


def scale_percent(problem):
    match = re.fullmatch(r"beam_assembly_step1_scale_(\d{3})", problem)
    if match is None:
        return None
    percent = int(match[1])
    if not 1 <= percent < 100:
        raise ValueError("Beam scale must be between 1 and 99 percent")
    return percent


def problem_name(percent):
    if not isinstance(percent, int) or not 1 <= percent < 100:
        raise ValueError('Beam scale must be an integer between 1 and 99 percent')
    name = f"beam_assembly_step1_scale_{percent:03d}"
    scale_percent(name)
    return name


def task_name(percent):
    return f"Isaacsimenvs-PreciseAssemblyBeamScale{percent:03d}-Direct-v0"


def register_scaled_beams(problems, objects, root):
    from evaluation.objects import Object

    original = problems['beam_assembly_step1']
    source = objects[original.insertion_object_name]
    for directory in sorted((root / 'assets/urdf/fabrica/beam_3x').glob('scale_[0-9][0-9][0-9]')):
        percent = int(directory.name[-3:])
        name = problem_name(percent)
        if not all((directory / file).is_file() for file in ('object.urdf', 'fixture.urdf', 'metadata.json')):
            raise FileNotFoundError(f'Incomplete scaled beam: {directory}')
        s = percent / 100
        objects[name] = Object(urdf_path=directory / 'object.urdf', scale=tuple(v * s for v in source.scale), need_vhacd=False)
        problems[name] = dataclasses.replace(
            original, name=name, insertion_object_name=name,
            receptive_urdf=str((directory / 'fixture.urdf').relative_to(root / 'assets')),
            insert_pose_rel_receptive=tuple(tuple(v * s for v in p[:3]) + p[3:] for p in original.insert_pose_rel_receptive),
            hole_z_offset=original.hole_z_offset * s,
            pre_insert_offset=original.pre_insert_offset * s,
            prelude_lift_offset=original.prelude_lift_offset * s,
        )
