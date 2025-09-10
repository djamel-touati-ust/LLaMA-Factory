import os
from typing import Any

from .common import export_datasets, save_args
from .locales import ALERTS

EXPORT_CONFIG_DIR = os.getenv("EXPORT_CONFIG_DIR", "/work/configs")
EXPORT_DATA_DIR = os.getenv("EXPORT_DATA_DIR", "/work/data")


class Runner:
    """Minimal runner that only saves configs and copies datasets."""

    def __init__(self, manager, demo_mode: bool = False) -> None:
        self.manager = manager
        self.demo_mode = demo_mode

    def _build_config_dict(self, data: dict) -> dict[str, Any]:
        config_dict: dict[str, Any] = {}
        skip_ids = ["top.lang", "top.model_path", "train.output_dir", "train.config_path"]
        for elem, value in data.items():
            elem_id = self.manager.get_id_by_elem(elem)
            if elem_id not in skip_ids:
                config_dict[elem_id] = value
        return config_dict

    def save_args(self, data):
        output_box = self.manager.get_elem_by_id("train.output_box")
        lang = data[self.manager.get_elem_by_id("top.lang")]
        config_path = data[self.manager.get_elem_by_id("train.config_path")]
        os.makedirs(EXPORT_CONFIG_DIR, exist_ok=True)
        save_path = os.path.join(EXPORT_CONFIG_DIR, config_path)
        save_args(save_path, self._build_config_dict(data))

        dataset_dir = data[self.manager.get_elem_by_id("train.dataset_dir")]
        datasets = data[self.manager.get_elem_by_id("train.dataset")]
        if dataset_dir and datasets:
            export_datasets(dataset_dir, datasets, EXPORT_DATA_DIR)

        return {output_box: ALERTS["info_config_saved"][lang] + save_path}
