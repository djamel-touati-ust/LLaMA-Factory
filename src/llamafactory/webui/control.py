import os
from typing import Optional

from ..extras.constants import (
    CHECKPOINT_NAMES,
    PEFT_METHODS,
    STAGES_USE_PAIR_DATA,
    TRAINING_STAGES,
)
from ..extras.packages import is_gradio_available
from .common import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_DATA_DIR,
    get_model_path,
    get_save_dir,
    get_template,
    load_dataset_info,
)
from .locales import ALERTS

if is_gradio_available():
    import gradio as gr


def switch_hub(hub_name: str) -> None:
    os.environ["USE_MODELSCOPE_HUB"] = "1" if hub_name == "modelscope" else "0"
    os.environ["USE_OPENMIND_HUB"] = "1" if hub_name == "openmind" else "0"


def can_quantize(finetuning_type: str) -> "gr.Dropdown":
    if finetuning_type not in PEFT_METHODS:
        return gr.Dropdown(value="none", interactive=False)
    else:
        return gr.Dropdown(interactive=True)


def can_quantize_to(quantization_method: str) -> "gr.Dropdown":
    if quantization_method == "bnb":
        available_bits = ["none", "8", "4"]
    elif quantization_method == "hqq":
        available_bits = ["none", "8", "6", "5", "4", "3", "2", "1"]
    elif quantization_method == "eetq":
        available_bits = ["none", "8"]
    else:
        available_bits = ["none"]
    return gr.Dropdown(choices=available_bits)


def change_stage(training_stage: str = list(TRAINING_STAGES.keys())[0]) -> tuple[list[str], bool]:
    return [], TRAINING_STAGES[training_stage] == "pt"


def get_model_info(model_name: str) -> tuple[str, str]:
    return get_model_path(model_name), get_template(model_name)


def check_template(lang: str, template: str) -> None:
    if template == "default":
        gr.Warning(ALERTS["warn_no_instruct"][lang])


def list_checkpoints(model_name: str, finetuning_type: str) -> "gr.Dropdown":
    checkpoints = []
    if model_name:
        save_dir = get_save_dir(model_name, finetuning_type)
        if save_dir and os.path.isdir(save_dir):
            for checkpoint in os.listdir(save_dir):
                if os.path.isdir(os.path.join(save_dir, checkpoint)) and any(
                    os.path.isfile(os.path.join(save_dir, checkpoint, name)) for name in CHECKPOINT_NAMES
                ):
                    checkpoints.append(checkpoint)
    if finetuning_type in PEFT_METHODS:
        return gr.Dropdown(value=[], choices=checkpoints, multiselect=True)
    else:
        return gr.Dropdown(value=None, choices=checkpoints, multiselect=False)


def list_config_paths(current_time: str) -> "gr.Dropdown":
    config_files = [f"{current_time}.yaml"]
    if os.path.isdir(DEFAULT_CONFIG_DIR):
        for file_name in os.listdir(DEFAULT_CONFIG_DIR):
            if file_name.endswith(".yaml") and file_name not in config_files:
                config_files.append(file_name)
    return gr.Dropdown(choices=config_files)


def list_datasets(dataset_dir: str = None, training_stage: str = list(TRAINING_STAGES.keys())[0]) -> "gr.Dropdown":
    dataset_info = load_dataset_info(dataset_dir if dataset_dir is not None else DEFAULT_DATA_DIR)
    ranking = TRAINING_STAGES[training_stage] in STAGES_USE_PAIR_DATA
    datasets = [k for k, v in dataset_info.items() if v.get("ranking", False) == ranking]
    return gr.Dropdown(choices=datasets)


def list_output_dirs(model_name: Optional[str], finetuning_type: str, current_time: str) -> "gr.Dropdown":
    output_dirs = [f"train_{current_time}"]
    if model_name:
        save_dir = get_save_dir(model_name, finetuning_type)
        if save_dir and os.path.isdir(save_dir):
            for folder in os.listdir(save_dir):
                output_dir = os.path.join(save_dir, folder)
                if os.path.isdir(output_dir):
                    output_dirs.append(folder)
    return gr.Dropdown(choices=output_dirs)
