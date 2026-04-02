import json
import os
import re
from typing import List, Optional, Tuple

from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.longcat_next import LongCatNext


@register_model("longcat_next_visual_cot", "longcat-next-visual-cot")
class LongCatNextVisualCoT(lmms):
    """
    LongCat-Next visual CoT.

    Stage 1: generate an auxiliary image from the question.
    Stage 2: answer with original image(s) + generated auxiliary image.
    """

    def __init__(
        self,
        pretrained: str = "meituan-longcat/LongCat-Next",
        generation_prompt_template: str = (
            "Generate a detailed visual diagram or illustration to help answer "
            "this question: {question}"
        ),
        output_dir: Optional[str] = None,
        save_intermediate: bool = False,
        fail_gracefully: bool = True,
        stage1_max_new_tokens: int = 2048,
        stage2_max_new_tokens: int = 1024,
        stage2_do_sample: bool = False,
        stage2_temperature: float = 0.0,
        system_prompt: str = "You are a helpful assistant.",
        **kwargs,
    ) -> None:
        super().__init__()

        self.pretrained = pretrained
        self.generation_prompt_template = generation_prompt_template
        self.save_intermediate = save_intermediate
        self.fail_gracefully = fail_gracefully
        self.stage1_max_new_tokens = stage1_max_new_tokens
        self.stage2_max_new_tokens = stage2_max_new_tokens
        self.stage2_do_sample = stage2_do_sample
        self.stage2_temperature = stage2_temperature

        self.output_dir = output_dir or "./logs/longcat_next_visual_cot"
        self.generated_images_dir = os.path.join(self.output_dir, "generated_images")
        self.intermediate_dir = os.path.join(self.output_dir, "intermediate_artifacts")
        os.makedirs(self.generated_images_dir, exist_ok=True)
        if self.save_intermediate:
            os.makedirs(self.intermediate_dir, exist_ok=True)

        self.longcat = LongCatNext(
            pretrained=pretrained,
            mode="generation",
            output_image_dir=self.generated_images_dir,
            response_persistent_folder=os.path.join(self.output_dir, "persistent"),
            continual_mode=False,
            system_prompt=system_prompt,
            **kwargs,
        )

    @property
    def rank(self):
        return self.longcat.rank

    @property
    def world_size(self):
        return self.longcat.world_size

    @property
    def model(self):
        return self.longcat.model

    @property
    def tokenizer(self):
        return self.longcat.tokenizer

    def _save_intermediate_artifacts(
        self,
        doc_id: str,
        task: str,
        generation_prompt: str,
        stage1_text: str,
        generated_images: List[str],
        question: str,
        final_answer: str,
    ) -> None:
        if not self.save_intermediate:
            return

        task_dir = os.path.join(self.intermediate_dir, task)
        os.makedirs(task_dir, exist_ok=True)
        metadata_path = os.path.join(task_dir, f"{doc_id}_metadata.json")
        metadata = {
            "doc_id": doc_id,
            "task": task,
            "generation_prompt": generation_prompt,
            "stage1_text": stage1_text,
            "generated_images": generated_images,
            "question": question,
            "stage2_answer": final_answer,
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def _stage1_generate_image(
        self,
        generation_prompt: str,
        doc_id: str,
        task: str,
        original_visuals: Optional[List] = None,
    ) -> Tuple[str, List[str]]:
        try:
            return self.longcat.generate_text_and_image(
                generation_prompt,
                doc_id=f"{doc_id}_stage1",
                task=task,
                visuals=original_visuals,
                gen_kwargs={"max_new_tokens": self.stage1_max_new_tokens},
                save_name=f"{task}_{doc_id}_stage1",
            )
        except Exception as exc:
            eval_logger.error(f"LongCat-Next stage 1 failed for doc {doc_id}: {exc}")
            if self.fail_gracefully:
                return "", []
            raise

    def _stage2_answer(
        self,
        question: str,
        generated_image_path: str,
        doc_id: str,
        task: str,
        original_visuals: Optional[List] = None,
    ) -> str:
        try:
            stage2_visuals = list(original_visuals or [])
            stage2_visuals.append(generated_image_path)
            return self.longcat.generate_text(
                question,
                visuals=stage2_visuals,
                gen_kwargs={
                    "max_new_tokens": self.stage2_max_new_tokens,
                    "do_sample": self.stage2_do_sample,
                    "temperature": self.stage2_temperature,
                },
                doc_id=f"{doc_id}_stage2",
                task=task,
            )
        except Exception as exc:
            eval_logger.error(f"LongCat-Next stage 2 failed for doc {doc_id}: {exc}")
            if self.fail_gracefully:
                return ""
            raise

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res: List[str] = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="LongCatNextVisualCoT Generating",
        )

        for request in requests:
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.args
            visuals = []
            if doc_to_visual is not None:
                try:
                    doc = self.task_dict[task][split][doc_id]
                    visuals = self.longcat.flatten([doc_to_visual(doc)])
                except Exception as exc:
                    eval_logger.warning(
                        f"Failed to extract original visuals for doc {doc_id}: {exc}"
                    )

            gen_prompt_match = re.search(
                r"\[GEN_PROMPT\](.*?)\[/GEN_PROMPT\]",
                contexts,
                re.DOTALL,
            )
            question_match = re.search(
                r"\[QUESTION\](.*?)\[/QUESTION\]",
                contexts,
                re.DOTALL,
            )
            if gen_prompt_match and question_match:
                actual_question = question_match.group(1).strip()
                generation_prompt = gen_prompt_match.group(1).strip().replace(
                    "{question}",
                    actual_question,
                )
                question = actual_question
            else:
                question = contexts
                generation_prompt = self.generation_prompt_template.format(
                    question=contexts
                )

            stage1_text, generated_images = self._stage1_generate_image(
                generation_prompt=generation_prompt,
                doc_id=str(doc_id),
                task=task,
                original_visuals=visuals if visuals else None,
            )

            if not generated_images:
                fallback = stage1_text if stage1_text else ""
                res.append(fallback)
                pbar.update(1)
                continue

            final_answer = self._stage2_answer(
                question=question,
                generated_image_path=generated_images[0],
                doc_id=str(doc_id),
                task=task,
                original_visuals=visuals if visuals else None,
            )

            self._save_intermediate_artifacts(
                doc_id=str(doc_id),
                task=task,
                generation_prompt=generation_prompt,
                stage1_text=stage1_text,
                generated_images=generated_images,
                question=question,
                final_answer=final_answer,
            )

            res.append(final_answer)
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError(
            "LongCatNextVisualCoT does not support loglikelihood."
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError(
            "Multi-round dialogue is not implemented for LongCatNextVisualCoT."
        )
