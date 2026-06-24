"""Simple spike-based image captioning on top of multimodal fusion."""

from __future__ import annotations

from itertools import islice

from torch import Tensor

from bioarn.training.text_training import TextGenerationTrainer

from .fusion import MultimodalFusion


class SpikeCaptioner:
    """Generate text descriptions of visual inputs using spike-based processing."""

    def __init__(self, fusion: MultimodalFusion, text_trainer: TextGenerationTrainer):
        self.fusion = fusion
        self.text_trainer = text_trainer

    def caption(self, image: Tensor) -> str:
        """Run image -> concept -> text seed -> generation."""

        seed = self.fusion.describe_image(image, max_words=3).strip()
        if not seed:
            seed = "image"
        if not self.text_trainer.token_counts:
            return seed
        prompt = seed[: max(1, min(4, len(seed)))]
        continuation = self.text_trainer.generate(
            prompt,
            max_tokens=max(4, self.fusion.config.max_description_length),
            temperature=0.8,
        ).strip()
        caption = f"{seed} {continuation}".strip()
        return " ".join(caption.split())

    def train_captioning(self, image_text_pairs, num_pairs: int = 100):
        """Online-train caption associations and a lightweight text prior."""

        captions: list[str] = []
        for image, caption in islice(image_text_pairs, max(0, int(num_pairs))):
            text = str(caption).strip()
            if not text:
                continue
            self.fusion.learn_cross_modal(image, text, label=text)
            captions.append(text)
        if captions:
            corpus = "\n".join(captions)
            self.text_trainer.train_on_text(corpus, context_length=min(32, max(8, len(max(captions, key=len)))))
        return {"trained_pairs": len(captions)}


__all__ = ["SpikeCaptioner"]
