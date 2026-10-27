"""Create a weights-only initializer with explicit normalization/exploration priors."""

import argparse
import copy
import json
from pathlib import Path

import torch


def prepare_weights(checkpoint, *, prior_count=100000., sigma_range=(.35, .10)):
    source = checkpoint.get(0, checkpoint)
    weights = copy.deepcopy({k: source[k] for k in ('model', 'assymetric_vf_nets', 'central_val_stats') if k in source})
    changes = []
    for section, state in weights.items():
        for key, value in state.items():
            if prior_count is not None and key.endswith('mean_std.count') and isinstance(value, torch.Tensor):
                changes.append({'key': section + '/' + key, 'old_count': float(value), 'new_count': prior_count})
                value.fill_(prior_count)
    if sigma_range is not None:
        sigma = weights['model']['a2c_network.sigma']
        if sigma.ndim != 2 or sigma.shape[0] != 6:
            raise ValueError('Expected a six-group SAPG log-standard-deviation matrix')
        high, low = sigma_range
        if not 0 < low <= high:
            raise ValueError('Require 0 < sigma_min <= sigma_max')
        ladder = torch.linspace(high, low, 6, dtype=sigma.dtype, device=sigma.device)
        sigma.copy_(ladder.log().unsqueeze(1).expand_as(sigma))
        changes.append({'key': 'model/a2c_network.sigma', 'action_std_by_group': ladder.tolist()})
    return weights, changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prior-count', type=float, default=100000.)
    parser.add_argument('--keep-rms-counts', action='store_true')
    parser.add_argument('--keep-sigma', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.prior_count <= 0:
        raise ValueError('Prior count must be positive')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    weights, changes = prepare_weights(checkpoint, prior_count=None if args.keep_rms_counts else args.prior_count,
                                       sigma_range=None if args.keep_sigma else (.35, .10))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights, args.output)
    args.output.with_suffix('.json').write_text(json.dumps({
        'source_checkpoint': str(args.checkpoint), 'weights_only_not_resumable': True,
        'changes': changes, 'geometry_or_success_threshold_changes': False,
    }, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
