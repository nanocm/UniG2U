import json
import os
import re
from copy import deepcopy
import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from accelerate import Accelerator
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

LONGCAT_IMAGE_START = "<longcat_img_start>"
LONGCAT_IMAGE_END = "<longcat_img_end>"
LONGCAT_AUDIO_START = "<longcat_audio_start>"
LONGCAT_AUDIO_END = "<longcat_audio_end>"
LONGCAT_IMAGE_GENERATION_TRIGGER = "<longcat_img_start>"
DEFAULT_ANYRES_PREFIX = "<longcat_img_token_size>{h} {w}</longcat_img_token_size>"


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


@register_model("longcat_next", "longcat-next")
class LongCatNext(lmms):
    """
    LongCat-Next integration for lmms-eval.

    Supports:
    - visual understanding
    - text-to-image and image-conditioned image generation
    - Uni-MMMU interleaved evaluation via bagel-compatible kwargs
    """

    def __init__(
        self,
        pretrained: str = "meituan-longcat/LongCat-Next",
        mode: str = "understanding",
        device: str = "cuda",
        device_map: Optional[str] = "auto",
        max_memory: Optional[str] = None,
        batch_size: Union[int, str] = 1,
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        attn_implementation: Optional[str] = None,
        max_new_tokens: int = 1024,
        do_sample: bool = False,
        temperature: float = 0.0,
        top_k: Optional[int] = 20,
        top_p: Optional[float] = 0.85,
        repetition_penalty: float = 1.1,
        visual_do_sample: bool = True,
        visual_temperature: float = 0.5,
        visual_top_k: int = 1024,
        visual_top_p: float = 0.75,
        visual_cfg_scale: float = 3.0,
        visual_token_h: int = 37,
        visual_token_w: int = 37,
        visual_anyres_prefix: str = DEFAULT_ANYRES_PREFIX,
        system_prompt: str = "You are a helpful assistant.",
        continual_mode: bool = True,
        response_persistent_folder: Optional[str] = None,
        output_image_dir: Optional[str] = None,
        prompt_asset_dir: Optional[str] = None,
        fix_lazy_decoder_paths: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        if mode not in ["understanding", "generation"]:
            raise ValueError(
                f"mode must be 'understanding' or 'generation', got '{mode}'"
            )

        self.pretrained = pretrained
        self.mode = mode
        self.batch_size_per_gpu = int(batch_size)
        self.trust_remote_code = trust_remote_code
        self.attn_implementation = attn_implementation
        self.fix_lazy_decoder_paths = fix_lazy_decoder_paths
        self.system_prompt = system_prompt
        self.max_memory = self._parse_max_memory(max_memory)

        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

        self.visual_do_sample = visual_do_sample
        self.visual_temperature = visual_temperature
        self.visual_top_k = visual_top_k
        self.visual_top_p = visual_top_p
        self.visual_cfg_scale = visual_cfg_scale
        self.visual_token_h = visual_token_h
        self.visual_token_w = visual_token_w
        self.visual_anyres_prefix = visual_anyres_prefix

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
            if device_map == "auto":
                eval_logger.warning(
                    "LongCat-Next is best used with a single process and "
                    "device_map=auto. Falling back to one GPU per process "
                    "because multiple Accelerate processes were detected."
                )
                self.device_map = f"cuda:{accelerator.local_process_index}"
            else:
                self.device_map = device_map or f"cuda:{accelerator.local_process_index}"
            if continual_mode:
                eval_logger.warning(
                    "Continual mode is disabled under distributed inference."
                )
                continual_mode = False
        else:
            self._device = torch.device(device if torch.cuda.is_available() else "cpu")
            self._rank = 0
            self._world_size = 1
            self.device_map = device_map if device_map else device

        dtype_map = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        self.torch_dtype = dtype_map.get(dtype.lower(), torch.bfloat16)

        if response_persistent_folder is None:
            self.response_persistent_folder = "./logs/longcat_next_persistent_folder"
        else:
            self.response_persistent_folder = response_persistent_folder

        if output_image_dir is None:
            self.output_image_dir = os.path.join(
                self.response_persistent_folder, "longcat_next_generated_images"
            )
        else:
            self.output_image_dir = output_image_dir

        if prompt_asset_dir is None:
            self.prompt_asset_dir = os.path.join(
                self.response_persistent_folder, "longcat_next_prompt_assets"
            )
        else:
            self.prompt_asset_dir = prompt_asset_dir

        os.makedirs(self.output_image_dir, exist_ok=True)
        os.makedirs(self.prompt_asset_dir, exist_ok=True)

        self.continual_mode = continual_mode
        self.response_cache: Dict[str, str] = {}
        self.cache_mode = "start"
        if self.continual_mode:
            os.makedirs(self.response_persistent_folder, exist_ok=True)
            self.response_persistent_file = os.path.join(
                self.response_persistent_folder, "longcat_next_response.json"
            )
            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r", encoding="utf-8") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
                eval_logger.info(
                    f"Loaded cache: {len(self.response_cache)} records"
                )

        self._pretrained_root: Optional[Path] = None
        self._load_model()

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def processor(self):
        return self._processor

    @property
    def model(self):
        return self._model

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    def _load_model(self) -> None:
        model_kwargs = {
            "dtype": self.torch_dtype,
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.max_memory is not None:
            model_kwargs["max_memory"] = self.max_memory
        if self.attn_implementation is not None:
            model_kwargs["attn_implementation"] = self.attn_implementation

        eval_logger.info(f"Loading LongCat-Next from {self.pretrained}")
        self._model = AutoModelForCausalLM.from_pretrained(
            self.pretrained,
            **model_kwargs,
        ).eval()

        tokenizer_kwargs = {"trust_remote_code": self.trust_remote_code}
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.pretrained,
                fix_mistral_regex=True,
                **tokenizer_kwargs,
            )
        except TypeError:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.pretrained,
                **tokenizer_kwargs,
            )

        self._processor = AutoProcessor.from_pretrained(
            self.pretrained,
            trust_remote_code=self.trust_remote_code,
        )
        self._model.text_tokenizer = self._tokenizer
        self._config = self._model.config

        if self.fix_lazy_decoder_paths:
            self._patch_lazy_decoder_paths()

    def _parse_max_memory(self, raw: Optional[str]) -> Optional[Dict[Union[int, str], str]]:
        if raw is None:
            return None
        value = raw.strip()
        if not value:
            return None

        # Accept Python dict strings, e.g. "{0: '39GiB', 1: '39GiB'}"
        if value.startswith("{"):
            parsed = ast.literal_eval(value)
            if not isinstance(parsed, dict):
                raise ValueError(f"max_memory must be a dict string, got: {raw}")
            normalized: Dict[Union[int, str], str] = {}
            for key, item in parsed.items():
                try:
                    normalized[int(key)] = str(item)
                except (TypeError, ValueError):
                    normalized[str(key)] = str(item)
            return normalized

        # Accept compact form:
        # "0:39GiB;1:39GiB;2:39GiB;3:39GiB"
        normalized = {}
        for part in value.split(";"):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                raise ValueError(f"Invalid max_memory segment: {part}")
            key, item = part.split(":", 1)
            key = key.strip()
            item = item.strip()
            try:
                normalized[int(key)] = item
            except ValueError:
                normalized[key] = item
        return normalized or None

    def _resolve_pretrained_root(self) -> Optional[Path]:
        if self._pretrained_root is not None:
            return self._pretrained_root

        local_path = Path(self.pretrained)
        if local_path.exists():
            self._pretrained_root = local_path.resolve()
            return self._pretrained_root

        try:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                self.pretrained,
                "config.json",
                repo_type="model",
            )
            self._pretrained_root = Path(config_path).resolve().parent
            return self._pretrained_root
        except Exception as exc:
            eval_logger.warning(
                f"Failed to resolve LongCat-Next pretrained root for lazy "
                f"decoder patching: {exc}"
            )
            return None

    def _patch_lazy_decoder_paths(self) -> None:
        root = self._resolve_pretrained_root()
        if root is None:
            return

        image_decoder_path = root / "image_decoder" / "image_decoder.safetensors"
        audio_vocoder_path = root / "cosy24k_vocoder" / "hift.pt"

        config_targets = [self.model.config]
        core_model = getattr(self.model, "model", None)
        if core_model is not None and getattr(core_model, "config", None) is not None:
            config_targets.append(core_model.config)

        for cfg in config_targets:
            try:
                visual_decoder_config = cfg.visual_config.visual_decoder_config
                current_visual_path = getattr(visual_decoder_config, "weight_path", "")
                if image_decoder_path.exists() and (
                    not current_visual_path
                    or "WEIGHT_PATH_TO_LONGCAT_NEXT" in current_visual_path
                    or not os.path.exists(current_visual_path)
                ):
                    visual_decoder_config.weight_path = str(image_decoder_path)

                audio_cfg = cfg.audio_config.cosy24kvocoder_config
                current_audio_path = getattr(audio_cfg, "weight_path", "")
                if audio_vocoder_path.exists() and (
                    not current_audio_path
                    or "WEIGHT_PATH_TO_LONGCAT_NEXT" in current_audio_path
                    or not os.path.exists(current_audio_path)
                ):
                    audio_cfg.weight_path = str(audio_vocoder_path)
            except Exception as exc:
                eval_logger.warning(
                    f"Failed to patch LongCat-Next lazy decoder paths: {exc}"
                )

    def flatten(self, input_list: Sequence[Any]) -> List[Any]:
        output: List[Any] = []
        for item in input_list:
            if isinstance(item, list):
                output.extend(self.flatten(item))
            else:
                output.append(item)
        return output

    def format_output(self, text: str, images: List[str]) -> str:
        return json.dumps({"text": text, "images": images}, ensure_ascii=False)

    def _get_uuid(self, task: str, split: str, doc_id: Union[str, int]) -> str:
        return f"{task}___{split}___{doc_id}"

    def _sanitize_text_prompt(self, prompt: str) -> str:
        if prompt is None:
            return ""
        return prompt.strip()

    def _to_rgb_image(self, image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, dict):
            if image.get("path"):
                return Image.open(image["path"]).convert("RGB")
            if image.get("bytes"):
                from io import BytesIO

                return Image.open(BytesIO(image["bytes"])).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image)}")

    def _materialize_image(
        self,
        image: Union[str, Image.Image, Dict[str, Any]],
        doc_id: str,
        task: str,
        name: str,
    ) -> str:
        if isinstance(image, str):
            image_path = Path(image).expanduser()
            if image_path.exists():
                return str(image_path.resolve())
            raise FileNotFoundError(f"Image path not found: {image}")

        image_dir = Path(self.prompt_asset_dir) / task
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{doc_id}_{name}.png"
        self._to_rgb_image(image).save(image_path)
        return str(image_path.resolve())

    def _extract_visuals(
        self,
        task: str,
        split: str,
        doc_id: Union[str, int],
        doc_to_visual,
    ) -> List[Any]:
        if doc_to_visual is None:
            return []
        doc = self.task_dict[task][split][doc_id]
        visuals = [doc_to_visual(doc)]
        return self.flatten(visuals)

    def _image_token(self, image_path: str) -> str:
        return f"{LONGCAT_IMAGE_START}{image_path}{LONGCAT_IMAGE_END}"

    def _audio_token(self, audio_path: str) -> str:
        return f"{LONGCAT_AUDIO_START}{audio_path}{LONGCAT_AUDIO_END}"

    def _render_content_items(
        self,
        content_items: Sequence[Any],
        doc_id: str,
        task: str,
    ) -> str:
        rendered_parts: List[str] = []
        for idx, item in enumerate(content_items):
            if item is None:
                continue

            if isinstance(item, str):
                text = item.strip()
                if text:
                    rendered_parts.append(text)
                continue

            if isinstance(item, dict):
                item_type = item.get("type")
                item_value = item.get("value")
                item_name = item.get("name", f"asset_{idx}")
                if item_type == "image":
                    image_path = self._materialize_image(
                        item_value, doc_id, task, item_name
                    )
                    rendered_parts.append(self._image_token(image_path))
                    continue
                if item_type == "audio":
                    audio_path = str(Path(item_value).expanduser().resolve())
                    rendered_parts.append(self._audio_token(audio_path))
                    continue

            raise TypeError(f"Unsupported content item: {item}")

        return "\n".join(rendered_parts)

    def _compose_prompt_with_visuals(
        self,
        prompt: str,
        visuals: Optional[Sequence[Any]],
        doc_id: str,
        task: str,
    ) -> str:
        prompt = self._sanitize_text_prompt(prompt)
        if not visuals:
            return prompt

        visual_tokens: List[str] = []
        for idx, visual in enumerate(visuals):
            image_path = self._materialize_image(
                visual,
                doc_id,
                task,
                f"input_{idx}",
            )
            visual_tokens.append(self._image_token(image_path))

        if "<image>" in prompt:
            prompt_with_visuals = prompt
            for visual_token in visual_tokens:
                prompt_with_visuals = prompt_with_visuals.replace(
                    "<image>",
                    visual_token,
                    1,
                )
            return prompt_with_visuals

        if re.search(r"<image \d+>", prompt):
            def _replace(match):
                image_idx = int(match.group(1)) - 1
                if image_idx < 0:
                    image_idx = 0
                if image_idx >= len(visual_tokens):
                    image_idx = len(visual_tokens) - 1
                return visual_tokens[image_idx]

            return re.sub(r"<image (\d+)>", _replace, prompt)

        parts = list(visual_tokens)
        if prompt:
            parts.append(prompt)
        return "\n".join(parts)

    def _prepare_inputs(self, messages: List[Dict[str, str]]):
        text_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        text_inputs, visual_inputs, audio_inputs = self.processor(
            text=text_input,
            return_tensors="pt",
        )
        text_inputs = text_inputs.to(self.model.device)
        if visual_inputs is not None:
            visual_inputs = visual_inputs.to(self.model.device)
        if audio_inputs is not None:
            audio_inputs = audio_inputs.to(self.model.device)
        return text_inputs, visual_inputs, audio_inputs

    def _base_generation_kwargs(self, gen_kwargs: Optional[Dict[str, Any]] = None):
        effective = gen_kwargs or {}
        do_sample = effective.get("do_sample", self.do_sample)
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": effective.get("max_new_tokens", self.max_new_tokens),
            "do_sample": do_sample,
            "repetition_penalty": effective.get(
                "repetition_penalty", self.repetition_penalty
            ),
        }
        if do_sample:
            generation_kwargs["temperature"] = effective.get(
                "temperature", self.temperature
            )
            if effective.get("top_k", self.top_k) is not None:
                generation_kwargs["top_k"] = effective.get("top_k", self.top_k)
            if effective.get("top_p", self.top_p) is not None:
                generation_kwargs["top_p"] = effective.get("top_p", self.top_p)
        else:
            temperature = effective.get("temperature", self.temperature)
            if temperature not in [None, 0, 0.0]:
                generation_kwargs["temperature"] = temperature

        return generation_kwargs

    def _build_visual_generation_config(
        self, gen_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        base = {
            "do_sample": self.visual_do_sample,
            "temperature": self.visual_temperature,
            "top_k": self.visual_top_k,
            "top_p": self.visual_top_p,
            "custom_params": {
                "cfg_scale": self.visual_cfg_scale,
                "token_h": self.visual_token_h,
                "token_w": self.visual_token_w,
                "anyres_prefix": self.visual_anyres_prefix,
            },
        }
        override = {}
        if gen_kwargs is not None:
            override = gen_kwargs.get("visual_generation_config", {})
        return _deep_update(base, override)

    def _run_messages(
        self,
        messages: List[Dict[str, str]],
        generation_kwargs: Dict[str, Any],
    ):
        text_inputs, visual_inputs, audio_inputs = self._prepare_inputs(messages)
        model_kwargs = {
            "input_ids": text_inputs["input_ids"],
            "return_dict_in_generate": True,
            **generation_kwargs,
        }
        if visual_inputs is not None:
            model_kwargs["visual_inputs"] = visual_inputs
        if audio_inputs is not None:
            model_kwargs["audio_inputs"] = audio_inputs

        with torch.no_grad():
            outputs = self.model.generate(**model_kwargs)
        return outputs, text_inputs

    def _decode_text_output(self, outputs, text_inputs) -> str:
        prompt_len = text_inputs["input_ids"].shape[1]
        generated_ids = outputs.sequences[0][prompt_len:]
        return self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

    def _decode_generated_images(
        self,
        outputs,
        save_prefix: str,
        gen_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        visual_ids = getattr(outputs, "visual_ids", None)
        if visual_ids is None or visual_ids.numel() == 0:
            return []

        visual_generation_config = self._build_visual_generation_config(gen_kwargs)
        custom_params = visual_generation_config.get("custom_params", {})
        core_model = getattr(self.model, "model", self.model)
        return core_model.decode_visual_ids_and_save(
            visual_ids,
            save_prefix=save_prefix,
            **custom_params,
        )

    def generate_text(
        self,
        prompt: str,
        visuals: Optional[Sequence[Any]] = None,
        gen_kwargs: Optional[Dict[str, Any]] = None,
        doc_id: str = "tmp",
        task: str = "tmp",
    ) -> str:
        content = self._compose_prompt_with_visuals(
            prompt,
            visuals,
            doc_id,
            task,
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content},
        ]
        outputs, text_inputs = self._run_messages(
            messages, self._base_generation_kwargs(gen_kwargs)
        )
        return self._decode_text_output(outputs, text_inputs)

    def generate_text_and_image(
        self,
        prompt: str,
        doc_id: str,
        task: str,
        visuals: Optional[Sequence[Any]] = None,
        gen_kwargs: Optional[Dict[str, Any]] = None,
        save_name: Optional[str] = None,
    ) -> Tuple[str, List[str]]:
        content = self._compose_prompt_with_visuals(
            prompt,
            visuals,
            doc_id,
            task,
        )
        if content:
            content = f"{content}{LONGCAT_IMAGE_GENERATION_TRIGGER}"
        else:
            content = LONGCAT_IMAGE_GENERATION_TRIGGER

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content},
        ]
        generation_kwargs = self._base_generation_kwargs(gen_kwargs)
        generation_kwargs["visual_generation_config"] = (
            self._build_visual_generation_config(gen_kwargs)
        )

        outputs, text_inputs = self._run_messages(messages, generation_kwargs)
        output_text = self._decode_text_output(outputs, text_inputs)
        image_save_prefix = os.path.join(
            self.output_image_dir,
            save_name or f"{task}_{doc_id}",
        )
        output_images = self._decode_generated_images(
            outputs,
            image_save_prefix,
            gen_kwargs,
        )
        return output_text, output_images

    def generate_from_content_items(
        self,
        content_items: Sequence[Any],
        doc_id: str,
        task: str,
        force_image_generation: bool = False,
        gen_kwargs: Optional[Dict[str, Any]] = None,
        save_name: Optional[str] = None,
    ) -> Tuple[str, List[str]]:
        content = self._render_content_items(content_items, doc_id, task)
        if force_image_generation:
            content = f"{content}{LONGCAT_IMAGE_GENERATION_TRIGGER}"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content},
        ]
        generation_kwargs = self._base_generation_kwargs(gen_kwargs)
        if force_image_generation:
            generation_kwargs["visual_generation_config"] = (
                self._build_visual_generation_config(gen_kwargs)
            )

        outputs, text_inputs = self._run_messages(messages, generation_kwargs)
        output_text = self._decode_text_output(outputs, text_inputs)
        output_images: List[str] = []
        if force_image_generation:
            output_images = self._decode_generated_images(
                outputs,
                os.path.join(
                    self.output_image_dir,
                    save_name or f"{task}_{doc_id}",
                ),
                gen_kwargs,
            )
        return output_text, output_images

    def generate_uni_mmmu_interleaved(
        self,
        input_images: List[Any],
        prompt: str,
        doc_id: str,
        task: str,
        interleaved_config: Dict[str, Any],
        doc: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, List[str]]:
        import json as json_module

        task_type = interleaved_config.get("task_type", "jigsaw")
        num_images = interleaved_config.get("num_images", 2)
        if doc is not None:
            if task_type == "maze":
                steps_str = doc.get("steps", "[]")
                steps = (
                    json_module.loads(steps_str)
                    if isinstance(steps_str, str)
                    else steps_str
                )
                if steps:
                    num_images = len(steps)
            elif task_type == "sliding":
                steps_str = doc.get("steps_words", "[]")
                steps = (
                    json_module.loads(steps_str)
                    if isinstance(steps_str, str)
                    else steps_str
                )
                if steps:
                    num_images = len(steps)

        base_items: List[Any] = []
        for idx, image in enumerate(input_images):
            base_items.append(
                {"type": "image", "value": image, "name": f"input_{idx}"}
            )
        base_items.append(self._sanitize_text_prompt(prompt))

        generated_images: List[str] = []

        if task_type == "jigsaw":
            suffix0 = (
                "Output ONLY a single image with Candidate 0 placed in the "
                "bottom-right cell. No text."
            )
            _, img0_paths = self.generate_from_content_items(
                [*base_items, suffix0],
                doc_id=doc_id,
                task=task,
                force_image_generation=True,
                gen_kwargs=interleaved_config,
                save_name=f"{task}_{doc_id}_cand0",
            )
            if img0_paths:
                generated_images.extend(img0_paths)

            suffix1 = (
                "Output ONLY a single image with Candidate 1 placed in the "
                "bottom-right cell. No text."
            )
            _, img1_paths = self.generate_from_content_items(
                [*base_items, suffix1],
                doc_id=doc_id,
                task=task,
                force_image_generation=True,
                gen_kwargs=interleaved_config,
                save_name=f"{task}_{doc_id}_cand1",
            )
            if img1_paths:
                generated_images.extend(img1_paths)

            final_items = list(base_items)
            if img0_paths:
                final_items.extend(
                    [
                        {"type": "image", "value": img0_paths[0], "name": "cand0_gen"},
                        "COMPLETED WITH CANDIDATE 0:",
                    ]
                )
            if img1_paths:
                final_items.extend(
                    [
                        {"type": "image", "value": img1_paths[0], "name": "cand1_gen"},
                        "COMPLETED WITH CANDIDATE 1:",
                    ]
                )
            final_items.append(
                'Now output EXACTLY ONE <FINAL_ANSWER_JSON>{"choice": 0 or 1, '
                '"rationale": "≤30 words"}</FINAL_ANSWER_JSON>\n'
                "Do not output any additional images."
            )
            final_text, _ = self.generate_from_content_items(
                final_items,
                doc_id=doc_id,
                task=task,
                force_image_generation=False,
                gen_kwargs=interleaved_config,
            )
            return final_text, generated_images

        history_items = list(base_items)
        for step_idx in range(1, num_images + 1):
            if task_type == "maze":
                plan_suffix = (
                    f'Now planning for step {step_idx}, Please output a sentence in '
                    'the form: "Next, move one step up/down/left/right."'
                )
            else:
                plan_suffix = (
                    f"Now planning for step {step_idx}, Please output a sentence "
                    "describing which tile to move and in which direction."
                )

            plan_text, _ = self.generate_from_content_items(
                [*history_items, plan_suffix],
                doc_id=doc_id,
                task=task,
                force_image_generation=False,
                gen_kwargs={"max_new_tokens": 128, **interleaved_config},
            )
            history_items.append(plan_text)

            step_suffix = f"Now, generate the image for step {step_idx}."
            _, step_image_paths = self.generate_from_content_items(
                [*history_items, step_suffix],
                doc_id=doc_id,
                task=task,
                force_image_generation=True,
                gen_kwargs=interleaved_config,
                save_name=f"{task}_{doc_id}_step_{step_idx:04d}",
            )
            if step_image_paths:
                generated_images.extend(step_image_paths)
                history_items.extend(
                    [
                        f"Image for step {step_idx}:",
                        {
                            "type": "image",
                            "value": step_image_paths[0],
                            "name": f"step_{step_idx:04d}",
                        },
                    ]
                )

        history_items.append(
            "After the images, emit EXACTLY ONE LINE containing ONLY the final "
            'move list as <ANSWER_JSON>[...]</ANSWER_JSON>. No other text.'
        )
        final_text, _ = self.generate_from_content_items(
            history_items,
            doc_id=doc_id,
            task=task,
            force_image_generation=False,
            gen_kwargs=interleaved_config,
        )
        return final_text, generated_images

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res: List[str] = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc=f"LongCat-Next ({self.mode})",
        )

        for request in requests:
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.args
            doc_uuid = self._get_uuid(task, split, doc_id)

            if self.continual_mode and self.cache_mode == "resume":
                cached_response = self.response_cache.get(doc_uuid)
                if cached_response:
                    res.append(cached_response)
                    pbar.update(1)
                    continue

            prompt = self._sanitize_text_prompt(contexts)
            effective_gen_kwargs = dict(gen_kwargs or {})
            longcat_interleaved = effective_gen_kwargs.get("longcat_interleaved")
            if longcat_interleaved is None:
                longcat_interleaved = effective_gen_kwargs.get("bagel_interleaved")

            visuals = self._extract_visuals(task, split, doc_id, doc_to_visual)

            if longcat_interleaved is not None:
                doc = self.task_dict[task][split][doc_id]
                output_text, output_images = self.generate_uni_mmmu_interleaved(
                    visuals,
                    prompt,
                    str(doc_id),
                    task,
                    longcat_interleaved,
                    doc,
                )
                formatted_output = self.format_output(output_text, output_images)
            elif self.mode == "generation":
                output_text, output_images = self.generate_text_and_image(
                    prompt,
                    str(doc_id),
                    task,
                    visuals=visuals if visuals else None,
                    gen_kwargs=effective_gen_kwargs,
                )
                formatted_output = self.format_output(output_text, output_images)
            else:
                output_text = self.generate_text(
                    prompt,
                    visuals=visuals if visuals else None,
                    gen_kwargs=effective_gen_kwargs,
                    doc_id=str(doc_id),
                    task=task,
                )
                formatted_output = output_text

            res.append(formatted_output)

            if self.continual_mode:
                self.response_cache[doc_uuid] = formatted_output
                with open(self.response_persistent_file, "w", encoding="utf-8") as f:
                    json.dump(self.response_cache, f, ensure_ascii=False, indent=2)

            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError(
            "LongCat-Next does not support loglikelihood in this integration."
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError(
            "Multi-round dialogue is not implemented for LongCat-Next."
        )
