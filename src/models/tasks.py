"""Per-dataset task specification.

Models and maintenance code are dataset-agnostic; everything dataset-specific a
task needs (which relation the link tasks live on, which node type carries the
label, the linkreg target key in the split file) is declared here and nowhere
else. `tasks` lists the subset that makes semantic sense for the dataset —
experiments must not force a task onto a dataset that doesn't list it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

EdgeType = Tuple[str, str, str]


@dataclass(frozen=True)
class TaskSpec:
    core_rel: EdgeType        # relation the link tasks are defined on
    label_nt: str             # node type carrying the classification label
    linkreg_target: str       # key of the regression target in split_linkreg
    tasks: List[str]


SPECS = {
    "finbench": TaskSpec(
        core_rel=("account", "transfer", "account"),
        label_nt="account",
        linkreg_target="y_log_amount",
        tasks=["linkpred", "linkreg", "nodelabel"],
    ),
    "faers": TaskSpec(
        core_rel=("drug", "causes", "term"),
        label_nt="drug",
        linkreg_target="y_log_count",
        tasks=["linkpred", "linkreg", "nodelabel"],
    ),
    "eicu": TaskSpec(
        core_rel=("stay", "received", "medication"),
        label_nt="stay",
        linkreg_target="y_log_dose",
        tasks=["linkpred", "linkreg", "nodelabel"],
    ),
}
# eicu_full is the credentialed full release; identical schema and tasks as eicu.
SPECS["eicu_full"] = SPECS["eicu"]

SPECS["snb"] = TaskSpec(
    core_rel=("person", "knows", "person"),
    label_nt="person",
    linkreg_target="",           # SNB has no continuous edge target
    tasks=["linkpred", "nodelabel"],
)


def spec(dataset: str) -> TaskSpec:
    if dataset not in SPECS:
        raise KeyError(f"no task spec for dataset '{dataset}'")
    return SPECS[dataset]
