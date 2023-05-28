# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from fairseq import metrics, utils, modules
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import ChoiceEnum
from fairseq.dataclass import FairseqDataclass

MASKED_POSITIONS_ONLY_CHOICES = ChoiceEnum(["none", "both", "lm"])


@dataclass
class T5MetroCriterionConfig(FairseqDataclass):
    discriminator_loss_weight: float = field(
        default=1.0,
        metadata={"help": "weight for discriminator loss"}
    )
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    masked_positions_only: MASKED_POSITIONS_ONLY_CHOICES = field(
        default="none",
        metadata={"help": "calculate loss on masked positions only"}
    )
    rtd_loss_weight: float = field(
        default=0.0,
        metadata={"help": "weight of the auxiliary binary classification loss"}
    )
    rtd_only: bool = field(
        default=False,
        metadata={"help": "compute rtd loss only"}
    )
    weighted_lm_loss: bool = field(
        default=False,
        metadata={"help": "use RTD prediction to weight LM loss"}
    )
    divide_by_mean_weight: bool = field(
        default=False,
        metadata={"help": "divide loss by mean loss weight; only valid when `weighte_lm_loss` is true"}
    )


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, loss_weight=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    nll_loss = nll_loss.squeeze(-1)
    smooth_loss = smooth_loss.squeeze(-1)
    if loss_weight is not None:
        nll_loss = nll_loss * loss_weight
        smooth_loss = smooth_loss * loss_weight
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion(
    "t5_metro", dataclass=T5MetroCriterionConfig
)
class T5MetroCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        discriminator_loss_weight,
        label_smoothing,
        masked_positions_only="none",
        rtd_loss_weight=0.0,
        rtd_only=False,
        weighted_lm_loss=False,
        divide_by_mean_weight=False,
    ):
        super().__init__(task)
        self.discriminator_loss_weight = discriminator_loss_weight
        self.eps = label_smoothing
        self.masked_positions_only = masked_positions_only
        self.rtd_loss_weight = rtd_loss_weight
        self.rtd_only = rtd_only
        self.weighted_lm_loss = weighted_lm_loss
        self.divide_by_mean_weight = divide_by_mean_weight

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        masked_tokens = sample["generator_target"].ne(self.padding_idx)
        generator_sample_size = masked_tokens.int().sum()

        # Rare: when all tokens are masked, project all tokens.
        # We use torch.where to avoid device-to-host transfers,
        # except on CPU where torch.where is not well supported
        # (see github.com/pytorch/pytorch/issues/26247).
        if masked_tokens.device == torch.device("cpu"):
            if not masked_tokens.any():
                masked_tokens = None
        else:
            masked_tokens = torch.where(
                masked_tokens.any(),
                masked_tokens,
                masked_tokens.new([True]),
            )

        discriminator_out, generator_out, replaced_input = model(
            **sample["net_input"],
            masked_tokens=masked_tokens
        )

        generator_target = sample["generator_target"]
        if masked_tokens is not None:
            generator_target = generator_target[masked_tokens]
        generator_logits = generator_out.view(-1, generator_out.size(-1))
        generator_target = generator_target.view(-1)
        generator_loss = modules.cross_entropy(
            generator_logits,
            generator_target,
            reduction="sum",
            ignore_index=self.padding_idx,
        )
        generator_padding_mask = generator_target.ne(self.padding_idx)
        with torch.no_grad():
            generator_preds = generator_logits.argmax(1)
        generator_n_correct = torch.sum(
            generator_preds.masked_select(generator_padding_mask)
                .eq(generator_target.masked_select(generator_padding_mask))
        )
        original_sentence = sample["discriminator_target"]
        replaced_tokens = (original_sentence.ne(self.padding_idx) & original_sentence.ne(replaced_input)).view(-1)
        unreplaced_tokens = (original_sentence.ne(self.padding_idx) & original_sentence.eq(replaced_input)).view(-1)
        valid_tokens = original_sentence.ne(self.padding_idx).view(-1)

        (
            discriminator_loss,
            discriminator_nll_loss,
            replaced_n_correct,
            replaced_total,
            unreplaced_n_correct,
            unreplaced_total,
            rtd_loss,
            rtd_replaced_n_correct,
            rtd_unreplaced_n_correct
        ) = self.compute_loss(
            model, discriminator_out, sample,
            masked_tokens.view(-1), replaced_tokens, unreplaced_tokens, valid_tokens,
            reduce=reduce
        )
        discriminator_sample_size = (
            generator_sample_size
            if self.masked_positions_only == "lm" or self.masked_positions_only == "both"
            else sample["sample_size"]
        )
        rtd_sample_size = (
            generator_sample_size
            if self.masked_positions_only == "both"
            else sample["sample_size"]
        )

        if self.rtd_only:
            discriminator_loss = discriminator_loss * torch.tensor(0.0)
        loss = (
            generator_loss / generator_sample_size * discriminator_sample_size
            + self.discriminator_loss_weight * discriminator_loss
            + self.rtd_loss_weight * rtd_loss / rtd_sample_size * discriminator_sample_size
        )
        logging_output = {
            "loss": loss.data,
            "generator_loss": generator_loss.data,
            "discriminator_loss": discriminator_loss.data,
            "discriminator_nll_loss": discriminator_nll_loss.data,
            "rtd_loss": rtd_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["discriminator_target"].size(0),
            "sample_size": discriminator_sample_size,
            "generator_sample_size": utils.item(generator_sample_size.data),
            "rtd_sample_size": utils.item(rtd_sample_size),
            "generator_n_correct": utils.item(generator_n_correct.data),
            "replaced_n_correct": utils.item(replaced_n_correct.data),
            "replaced_total": utils.item(replaced_total.data),
            "unreplaced_n_correct": utils.item(unreplaced_n_correct.data),
            "unreplaced_total": utils.item(unreplaced_total.data),
            "rtd_replaced_n_correct": utils.item(rtd_replaced_n_correct.data),
            "rtd_unreplaced_n_correct": utils.item(rtd_unreplaced_n_correct.data),
        }
        return loss, discriminator_sample_size, logging_output

    @staticmethod
    def get_lprobs_and_target(model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, masked_tokens,
                     replaced_tokens, unreplaced_tokens, valid_tokens, reduce=True):
        # compute discriminator loss
        # 1. types of all tokens:
        #    |                              all_tokens                          |
        #    |                       valid_tokens                    | paddings |
        #    |           masked_tokens             | unmasked_tokens | paddings |
        #    | replaced_tokens |                   | unmasked_tokens | paddings |
        #    | replaced_tokens |          unreplaced_tokens          | paddings |
        #
        #    valid_tokens  <==>  original_sentence != padding_id
        #    masked_tokens  <==>  generator_target != padding_id
        #
        # 2. lm loss
        #    `net_output` vs `discriminator_target`
        #    if `masked_positions_only` is true: compute on `masked_tokens`
        #    if `masked_positions_only` is false: compute on `valid_tokens`
        #    metrics:
        #     - `replaced_accuracy`: accuracy on `replaced_tokens`
        #     - `unreplaced_accuracy`: accuracy on `unreplaced_tokens`
        #       - WARNING: may include positions where loss is not computed (`masked_positions_only`)
        #
        # 3. rtd loss
        #    compute when `rtd_loss_weight` > 0.0
        #    `rtd_output` vs `replaced_tokens`
        #    if `masked_positions_only` is true: compute on `masked_tokens`
        #    if `masked_positions_only` is false: compute on `valid_tokens`
        #    metrics:
        #     - `rtd_replaced_accuracy`: accuracy on `replaced_tokens` (recall)
        #     - `rtd_unreplaced_accuracy`: accuracy on `unreplaced_tokens` (specificity)
        #       - WARNING: may include positions where loss is not computed (`masked_positions_only`)

        if self.rtd_loss_weight > 0.0:
            net_output, rtd_output = net_output
            rtd_output = rtd_output.view(-1).float()
            rtd_target = replaced_tokens.float()
            if self.masked_positions_only == "both":
                rtd_output_with_loss = rtd_output.masked_select(masked_tokens)
                rtd_target_with_loss = rtd_target.masked_select(masked_tokens)
            else:
                rtd_output_with_loss = rtd_output.masked_select(valid_tokens)
                rtd_target_with_loss = rtd_target.masked_select(valid_tokens)
            rtd_loss = F.binary_cross_entropy_with_logits(
                rtd_output_with_loss,
                rtd_target_with_loss,
                reduction="sum" if reduce else "none"
            )
            with torch.no_grad():
                rtd_preds = rtd_output.ge(0.0)
            n_correct_rtd_replaced = torch.sum(
                rtd_preds.masked_select(replaced_tokens)
                .eq(rtd_target.masked_select(replaced_tokens))
            )
            n_correct_rtd_unreplaced = torch.sum(
                rtd_preds.masked_select(unreplaced_tokens)
                .eq(rtd_target.masked_select(unreplaced_tokens))
            )
        else:
            rtd_output = None
            rtd_loss = torch.tensor(0.0, dtype=torch.float)
            n_correct_rtd_replaced = torch.tensor(0, dtype=torch.long)
            n_correct_rtd_unreplaced = torch.tensor(0, dtype=torch.long)

        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        if self.weighted_lm_loss:
            assert rtd_output is not None
            with torch.no_grad():
                lm_loss_weight = torch.sigmoid(rtd_output)
        else:
            lm_loss_weight = None
        if self.masked_positions_only == "both" or self.masked_positions_only == "lm":
            lprobs_with_loss = lprobs[masked_tokens, :]
            target_with_loss = target.masked_select(masked_tokens)
            if lm_loss_weight is not None:
                lm_loss_weight = lm_loss_weight.masked_select(masked_tokens)
        else:
            lprobs_with_loss = lprobs
            target_with_loss = target
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs_with_loss,
            target_with_loss,
            self.eps,
            ignore_index=self.padding_idx,
            loss_weight=lm_loss_weight,
            reduce=reduce,
        )

        if lm_loss_weight is not None:
            mean_loss_weight = lm_loss_weight.mean()
            if self.divide_by_mean_weight:
                loss = loss / mean_loss_weight
            nll_loss = nll_loss / mean_loss_weight

        with torch.no_grad():
            preds = lprobs.argmax(1)
        n_correct_replaced = torch.sum(
            preds.masked_select(replaced_tokens)
            .eq(target.masked_select(replaced_tokens))
        )
        total_replaced = torch.sum(replaced_tokens)
        n_correct_unreplaced = torch.sum(
            preds.masked_select(unreplaced_tokens)
            .eq(target.masked_select(unreplaced_tokens))
        )
        total_unreplaced = torch.sum(unreplaced_tokens)

        return (
            loss, nll_loss,
            n_correct_replaced, total_replaced, n_correct_unreplaced, total_unreplaced,
            rtd_loss, n_correct_rtd_replaced, n_correct_rtd_unreplaced
        )

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        generator_loss_sum = sum(log.get("generator_loss", 0) for log in logging_outputs)
        discriminator_loss_sum = sum(log.get("discriminator_loss", 0) for log in logging_outputs)
        discriminator_nll_loss_sum = sum(log.get("discriminator_nll_loss", 0) for log in logging_outputs)
        rtd_loss_sum = sum(log.get("rtd_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        discriminator_sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        generator_sample_size = utils.item(sum(log.get("generator_sample_size", 0) for log in logging_outputs))
        rtd_sample_size = utils.item(sum(log.get("rtd_sample_size", 0) for log in logging_outputs))
        generator_n_correct = utils.item(sum(log.get("generator_n_correct", 0) for log in logging_outputs))
        replaced_n_correct = utils.item(sum(log.get("replaced_n_correct", 0) for log in logging_outputs))
        replaced_total = utils.item(sum(log.get("replaced_total", 0) for log in logging_outputs))
        unreplaced_n_correct = utils.item(sum(log.get("unreplaced_n_correct", 0) for log in logging_outputs))
        unreplaced_total = utils.item(sum(log.get("unreplaced_total", 0) for log in logging_outputs))
        rtd_replaced_n_correct = utils.item(sum(log.get("rtd_replaced_n_correct", 0) for log in logging_outputs))
        rtd_unreplaced_n_correct = utils.item(sum(log.get("rtd_unreplaced_n_correct", 0) for log in logging_outputs))

        metrics.log_scalar(
            "loss", loss_sum / discriminator_sample_size / math.log(2), discriminator_sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", discriminator_nll_loss_sum / discriminator_sample_size / math.log(2),
            discriminator_sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )
        metrics.log_scalar(
            "generator_loss", generator_loss_sum / generator_sample_size / math.log(2), generator_sample_size, round=3
        )
        metrics.log_scalar(
            "discriminator_loss", discriminator_loss_sum / discriminator_sample_size / math.log(2),
            discriminator_sample_size, round=3
        )
        metrics.log_scalar(
            "rtd_loss", rtd_loss_sum / rtd_sample_size / math.log(2),
            rtd_sample_size, round=3
        )

        def _log_accuracy(key, key2=None):
            if key2 is None:
                key2 = key
            metrics.log_derived(
                f"{key}_accuracy",
                lambda meters: round(
                    meters[f"{key}_n_correct"].sum * 100.0 / meters[f"{key2}_total"].sum, 3
                )
                if meters[f"{key2}_total"].sum > 0
                else float("nan"),
            )

        metrics.log_scalar("generator_n_correct", generator_n_correct)
        metrics.log_scalar("generator_total", generator_sample_size)
        _log_accuracy("generator")

        metrics.log_scalar("replaced_n_correct", replaced_n_correct)
        metrics.log_scalar("replaced_total", replaced_total)
        _log_accuracy("replaced")

        metrics.log_scalar("unreplaced_n_correct", unreplaced_n_correct)
        metrics.log_scalar("unreplaced_total", unreplaced_total)
        _log_accuracy("unreplaced")

        metrics.log_scalar("rtd_replaced_n_correct", rtd_replaced_n_correct)
        _log_accuracy("rtd_replaced", "replaced")

        metrics.log_scalar("rtd_unreplaced_n_correct", rtd_unreplaced_n_correct)
        _log_accuracy("rtd_unreplaced", "unreplaced")

        metrics.log_derived(
            "replace_ratio",
            lambda meters: round(
                meters["replaced_total"].sum * 100.0 / (meters["replaced_total"].sum + meters["unreplaced_total"].sum),
                3
            )
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
