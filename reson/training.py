from art.trajectories import TrajectoryGroup, Trajectory
from art.skypilot import SkyPilotBackend
from reson.reson_base import ResonBase
from pydantic import PrivateAttr, Field
from pathlib import Path
import enum
import uuid
from typing import Optional

import sky


class TrainingManager(ResonBase):
    _instance: "TrainingManager" = PrivateAttr()
    counter: int = Field(default=0)
    epochs: int = Field(default=0)
    trajectory_groups: dict[int, TrajectoryGroup] = Field(default_factory=dict)
    trajectories: dict[str, Trajectory] = Field(default_factory=dict)
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    name: str

    def __new__(cls, **_kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance

    @classmethod
    def load(cls, path: Path | str) -> "TrainingManager":
        with open(path, "r") as f:
            contents = f.read()
            cls._instance = TrainingManager.model_validate_json(contents)

        return cls._instance

    @classmethod
    def save(cls, path: Path | str):

        if not cls._instance:
            raise ValueError("No instance of training manager set.")

        contents = cls._instance.model_dump_json()

        with open(path, "w") as f:
            f.write(contents)

    def new_group(self):
        self.trajectory_groups[self.counter] = TrajectoryGroup([])
        self.counter += 1

        return self.trajectory_groups[self.counter - 1]

    def new_trajectory(self, key: str):
        self.trajectories[key] = Trajectory()

        return self.trajectories[key]


__all__ = ["Trajectory", "TrajectoryGroup", "SkyPilotBackend", "sky", "TrainingManager"]
