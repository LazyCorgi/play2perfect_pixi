"""Prepare half-size beam step1 as an independent fallback experiment."""

import json

from generate_half_scale_insertion import ROOT, scaled_urdf


def main():
    root = ROOT / 'assets/urdf/fabrica/beam_3x'
    output = root / 'half_scale'
    output.mkdir(exist_ok=True)
    for source, name in (
        (root / '2/sdf_hybrid/beam_3x_2_matchedmass_sdf_hybrid.urdf', 'beam_step1_half.urdf'),
        (root / 'insertion_fixtures/part_2_sdf_hybrid.urdf', 'beam_step1_fixture_half.urdf'),
    ):
        target = output / name
        scaled_urdf(source, .5, destination=target).write(target, encoding='utf-8', xml_declaration=True)
        print(target.relative_to(ROOT))
    (output / 'metadata.json').write_text(json.dumps({
        'problem': 'beam_assembly_step1_half', 'source_problem': 'beam_assembly_step1',
        'linear_scale': .5, 'mass_scale': .125, 'inertia_scale': .03125,
        'mass_assumption': 'constant density; hardware not validated',
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
