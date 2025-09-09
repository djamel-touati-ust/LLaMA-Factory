# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
from collections.abc import Generator
from typing import TYPE_CHECKING

from ...extras.packages import is_gradio_available
from ..common import DEFAULT_CONFIG_DIR, DEFAULT_DATA_DIR, get_time, load_dataset_info, save_args
from ..locales import ALERTS
from .data import create_preview_box

if is_gradio_available():
    import gradio as gr

if TYPE_CHECKING:
    from gradio.components import Component
    from ..engine import Engine


def list_datasets(dataset_dir: str = None) -> "gr.Dropdown":
    dataset_info = load_dataset_info(dataset_dir if dataset_dir is not None else DEFAULT_DATA_DIR)
    return gr.Dropdown(choices=list(dataset_info.keys()))


def export_config_and_dataset(
    lang: str,
    model_name: str,
    model_path: str,
    finetuning_type: str,
    dataset_dir: str,
    dataset: list[str],
) -> Generator[str, None, None]:
    error = ""
    if not model_name:
        error = ALERTS["err_no_model"][lang]
    elif not model_path:
        error = ALERTS["err_no_path"][lang]
    elif not dataset:
        error = ALERTS["err_no_dataset"][lang]

    if error:
        gr.Warning(error)
        yield error
        return

    os.makedirs(DEFAULT_CONFIG_DIR, exist_ok=True)
    os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)

    config_path = os.path.join(DEFAULT_CONFIG_DIR, f"{get_time()}.yaml")
    config = dict(
        model_name=model_name,
        model_path=model_path,
        finetuning_type=finetuning_type,
        dataset=dataset,
    )
    save_args(config_path, config)

    dataset_info = load_dataset_info(dataset_dir)
    for name in dataset:
        file_name = dataset_info.get(name, {}).get("file_name")
        if not file_name:
            continue
        src_path = os.path.join(dataset_dir, file_name)
        dst_path = os.path.join(DEFAULT_DATA_DIR, os.path.basename(file_name))
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        elif os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)

    yield ALERTS["info_exporting"][lang]
    yield ALERTS["info_exported"][lang]


def create_export_tab(engine: "Engine") -> dict[str, "Component"]:
    with gr.Row():
        dataset_dir = gr.Textbox(value=DEFAULT_DATA_DIR, scale=1)
        dataset = gr.Dropdown(multiselect=True, allow_custom_value=True, scale=4)
        preview_elems = create_preview_box(dataset_dir, dataset)

    dataset_dir.change(list_datasets, [dataset_dir], [dataset], queue=False)
    dataset.focus(list_datasets, [dataset_dir], [dataset], queue=False)

    export_btn = gr.Button()
    info_box = gr.Textbox(show_label=False, interactive=False)

    export_btn.click(
        export_config_and_dataset,
        [
            engine.manager.get_elem_by_id("top.lang"),
            engine.manager.get_elem_by_id("top.model_name"),
            engine.manager.get_elem_by_id("top.model_path"),
            engine.manager.get_elem_by_id("top.finetuning_type"),
            dataset_dir,
            dataset,
        ],
        [info_box],
    )

    return dict(
        dataset_dir=dataset_dir,
        dataset=dataset,
        export_btn=export_btn,
        info_box=info_box,
        **preview_elems,
    )
