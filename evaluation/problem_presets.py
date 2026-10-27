"""Shared task/tolerance selection for published tasks and half-size insertion."""

DEFAULT_TASK = "Isaacsimenvs-PreciseAssembly-Direct-v0"
HALF_TASK = "Isaacsimenvs-PreciseAssemblyHalf-Direct-v0"
HALF_PROBLEMS = {"tight_insertion_half", "tight_insertion_half_original_mass", "beam_assembly_step1_half", "tight_insertion_075"}
CURRICULUM_TASK = "Isaacsimenvs-PreciseAssembly075-Direct-v0"


def default_goal_mode(problem: str) -> str:
    return "finalGoalOnly" if problem in HALF_PROBLEMS else "preInsertAndFinal"


def task_for_problem(problem: str, requested_task: str = DEFAULT_TASK) -> str:
    if problem == "tight_insertion_075" and requested_task in (DEFAULT_TASK, HALF_TASK):
        return CURRICULUM_TASK
    if requested_task == HALF_TASK and problem not in HALF_PROBLEMS:
        raise ValueError("The half-size task preset requires a registered half-size problem")
    if problem in HALF_PROBLEMS and requested_task == DEFAULT_TASK:
        return HALF_TASK
    return requested_task


def default_tolerances(problem: str) -> tuple[float, float]:
    # Must match the respective task YAML. Values describe keypoint proximity,
    # not mechanical clearance or a verified insertion.
    if problem == "tight_insertion_075":
        return .0075, .00375
    return (0.005, 0.0025) if problem in HALF_PROBLEMS else (0.01, 0.005)
