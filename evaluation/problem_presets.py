"""Shared problem selection and size-dependent success tolerances."""
from evaluation.beam_scaling import scale_percent, task_name
DEFAULT_TASK = "Isaacsimenvs-PreciseAssembly-Direct-v0"

def default_goal_mode(problem):
    return "preInsertAndFinal"

def task_for_problem(problem, requested_task=DEFAULT_TASK):
    percent = scale_percent(problem)
    if percent is None:
        return requested_task
    expected = task_name(percent)
    if requested_task not in (DEFAULT_TASK, expected):
        raise ValueError(f"{problem} requires {expected}")
    return expected

def default_tolerances(problem):
    percent = scale_percent(problem)
    scale = percent / 100 if percent is not None else 1
    return .01 * scale, .005 * scale
