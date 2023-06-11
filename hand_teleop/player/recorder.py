import pickle
from pathlib import Path
from typing import List, Callable, Dict, Any

import sapien.core as sapien
import time


class DataRecorder(object):
    def __init__(self, filename, scene: sapien.Scene):
        self.data_list = []
        self.recording_field_names: List[str] = []
        self.recording_fn_registry: List[Callable] = []
        self.all_field_names: List[str] = []
        self.filename = filename
        self.scene = scene

    def register_recording_fn(self, field_name: str, recoding_fn: Callable):
        if field_name in self.recording_field_names:
            raise ValueError(f"Duplicated filed name: {field_name}")
        self.recording_field_names.append(field_name)
        self.recording_fn_registry.append(recoding_fn)
        self.all_field_names.append(field_name)

    def generate_meta_data(self, additional_meta_data: Dict[str, Any]):
        meta_data = additional_meta_data.copy()
        timestamp = time.strftime('%m-%d_%H-%M-%S')
        data_len = len(self.data_list)
        registered_data_field = self.recording_field_names
        all_data_field = self.all_field_names

        # Actor and articulation information
        actor_dict = {actor.get_name(): actor.get_id() for actor in self.scene.get_all_actors()}
        all_articulation_root = [robot.get_links()[0] for robot in self.scene.get_all_articulations()]
        articulation_dict = {actor.get_name(): actor.get_id() for actor in all_articulation_root}
        articulation_dof = {r.get_links()[0].get_name(): r.dof for r in self.scene.get_all_articulations()}

        meta_data.update(dict(timestamp=timestamp, data_len=data_len, registered_data_field=registered_data_field,
                              all_data_field=all_data_field, actor=actor_dict, articulation=articulation_dict,
                              articulation_dof=articulation_dof, timestep=self.scene.get_timestep()))
        return meta_data

    def step(self, additional_data: Dict[str, Any]):
        for field_name, value in additional_data.items():
            if field_name in self.recording_field_names:
                raise ValueError(f"Duplicated filed name: {field_name}")
            if field_name not in self.all_field_names:
                self.all_field_names.append(field_name)
        data = additional_data.copy()
        data["simulation"] = self.scene.pack()
        for field, fn in zip(self.recording_field_names, self.recording_fn_registry):
            data[field] = fn()
        self.data_list.append(data)

    def dump(self, additional_meta_data: Dict[str, Any]):
        meta_data = self.generate_meta_data(additional_meta_data)
        dump = dict(meta_data=meta_data, data=self.data_list)
        root = Path(self.filename).parent
        root.mkdir(parents=True, exist_ok=True)
        with open(self.filename, "wb") as f:
            pickle.dump(dump, f)

    def __len__(self):
        return len(self.data_list)

    def clear(self):
        self.all_field_names.clear()
        self.recording_fn_registry.clear()
        self.recording_field_names.clear()
        self.data_list.clear()
