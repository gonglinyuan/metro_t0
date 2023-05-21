# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is ported from fairseq:
# https://github.com/facebookresearch/fairseq
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of the fairseq repo

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SentencepieceConfig:
    sentencepiece_model: str = field(
        default="???", metadata={"help": "path to sentencepiece model"}
    )
    sentencepiece_enable_sampling: bool = field(
        default=False, metadata={"help": "enable sampling"}
    )
    sentencepiece_alpha: Optional[float] = field(
        default=None, metadata={
            "help": "soothing parameter for unigram sampling, "
                    "and merge probability for BPE-dropout"
        }
    )


class SentencepieceBPE(object):
    def __init__(self, cfg):
        self.enable_sampling = cfg.sentencepiece_enable_sampling
        self.alpha = cfg.sentencepiece_alpha
        sentencepiece_model = cfg.sentencepiece_model
        try:
            import sentencepiece as spm

            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(sentencepiece_model)
        except ImportError:
            raise ImportError(
                "Please install sentencepiece with: pip install sentencepiece"
            )

    def encode(self, x: str) -> str:
        return " ".join(
            self.sp.Encode(
                x, out_type=str, enable_sampling=self.enable_sampling, alpha=self.alpha
            )
        )

    def decode(self, x: str) -> str:
        return x.replace(" ", "").replace("\u2581", " ").strip()

    def is_beginning_of_word(self, x: str) -> bool:
        if x in ["<unk>", "<s>", "</s>", "<pad>"]:
            # special elements are always considered beginnings
            # HACK: this logic is already present in fairseq/tasks/masked_lm.py
            # but these special tokens are also contained in the sentencepiece
            # vocabulary which causes duplicate special tokens. This hack makes
            # sure that they are all taken into account.
            return True
        return x.startswith("\u2581")
