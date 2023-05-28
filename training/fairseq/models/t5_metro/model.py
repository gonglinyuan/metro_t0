from dataclasses import field, dataclass

import torch
import torch.nn as nn

from fairseq.dataclass import ChoiceEnum
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.models import (
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture
)
from fairseq.models.roberta import RobertaEncoder, RobertaLMHead
from fairseq.models.transformer import (
    TransformerConfig,
    TransformerModel,
)
from fairseq.models.transformer.transformer_t5 import transformer_t5_base_rpe, transformer_t5_large_rpe
from fairseq.utils import safe_getattr

GENERATOR_SAMPLE_MODE_CHOICES = ChoiceEnum(["train", "eval"])
RTD_HEAD_CHOICES = ChoiceEnum(["none", "simple", "roberta"])
SHARE_GENERATOR_DISCRIMINATOR_EMBED_CHOICES = ChoiceEnum(["none", "token", "token_pos", "token_pos_norm"])


@dataclass
class T5MetroConfig(TransformerConfig):
    no_generator: bool = field(
        default=False, metadata={"help": "whether to build generator model"}
    )
    generator_sample_mode: GENERATOR_SAMPLE_MODE_CHOICES = field(
        default="train", metadata={"help": "which mode the generator is in when sampling from its MLM output"}
    )
    generator_zero_dropout: int = field(
        default=False, metadata={"help": "set generator dropout to zero"}
    )
    generator_layers: int = field(
        default=3, metadata={"help": "number of layers"}
    )
    rtd_head: RTD_HEAD_CHOICES = field(
        default="none", metadata={"help": "compute auxiliary RTD head"}
    )
    share_generator_discriminator_embed: SHARE_GENERATOR_DISCRIMINATOR_EMBED_CHOICES = field(
        default=False, metadata={"help": "share the embedding layer between the generator and the discriminator"}
    )


@register_model("t5_metro")
class T5MetroModel(FairseqEncoderDecoderModel):
    def __init__(self, cfg: T5MetroConfig, generator: RobertaEncoder, discriminator: TransformerModel):
        super(T5MetroModel, self).__init__(discriminator.encoder, discriminator.decoder)
        self.cfg = cfg
        self.generator = generator
        if self.cfg.rtd_head == "simple":
            self.rtd_head = nn.Linear(discriminator.decoder.output_embed_dim, 1)
            self.rtd_head.weight.data.zero_()
            self.rtd_head.bias.data.zero_()
        elif self.cfg.rtd_head == "roberta":
            self.rtd_head = RobertaLMHead(discriminator.decoder.output_embed_dim, 1, self.cfg.activation_fn)
            self.rtd_head.weight.data.zero_()
            self.rtd_head.bias.data.zero_()

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, T5MetroConfig(), delete_default=False, with_prefix=""
        )

    @classmethod
    def build_model(cls, args, task):
        discriminator = TransformerModel.build_model(args, task, T5MetroConfig)
        cfg = discriminator.cfg
        args.encoder_layers = args.generator_layers
        if args.generator_zero_dropout:
            args.attention_dropout = 0.0
            args.activation_dropout = 0.0
            args.dropout = 0.0
        generator = RobertaEncoder(args, task.source_dictionary)
        share_generator_discriminator_embed = set(cfg.share_generator_discriminator_embed.split("_"))
        if "token" in share_generator_discriminator_embed:
            generator.sentence_encoder.embed_tokens.weight = discriminator.encoder.embed_tokens.weight
            generator.lm_head.weight = discriminator.encoder.embed_tokens.weight
        if "pos" in share_generator_discriminator_embed:
            generator.sentence_encoder.embed_positions.weight = discriminator.encoder.embed_positions.weight
        if "norm" in share_generator_discriminator_embed:
            generator.sentence_encoder.layernorm_embedding.weight = discriminator.encoder.layernorm_embedding.weight
            generator.sentence_encoder.layernorm_embedding.bias = discriminator.encoder.layernorm_embedding.bias
        return cls(cfg, generator, discriminator)

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        masked_tokens=None
    ):
        if not self.cfg.no_generator:
            used_eval_mode = False
            if self.training and self.cfg.generator_sample_mode == "eval":
                self.generator.eval()
                with torch.no_grad():
                    generator_out_eval, _ = self.generator(
                        src_tokens,
                        features_only=False,
                        return_all_hiddens=False,
                        masked_tokens=masked_tokens
                    )
                self.generator.train()
                used_eval_mode = True
            generator_out, _ = self.generator(
                src_tokens,
                features_only=False,
                return_all_hiddens=False,
                masked_tokens=masked_tokens
            )
            if not used_eval_mode:
                generator_out_eval = generator_out.detach()

            with torch.no_grad():
                sample_probs = generator_out_eval.view(-1, generator_out_eval.size(-1))
                sample_probs = torch.softmax(sample_probs, -1, dtype=torch.float32)
                sampled_input = torch.multinomial(sample_probs, 1).view(-1)
                src_tokens = src_tokens.clone()
                src_tokens[masked_tokens] = sampled_input

        discriminator_encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        discriminator_features, discriminator_extra = self.decoder(
            prev_output_tokens,
            encoder_out=discriminator_encoder_out,
            features_only=True,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        discriminator_x = (
            discriminator_features
            if features_only
            else self.decoder.output_layer(discriminator_features)
        )
        discriminator_out = discriminator_x, discriminator_extra
        if self.cfg.rtd_head != "none":
            discriminator_encoder_features = discriminator_encoder_out["encoder_out"][0].transpose(0, 1)
            discriminator_out = (
                discriminator_out,
                self.rtd_head(discriminator_encoder_features).squeeze(-1)
            )

        if self.cfg.no_generator:
            return discriminator_out
        else:
            return discriminator_out, generator_out, src_tokens

    def get_targets(self, sample, net_output):
        return sample["discriminator_target"]


@register_model_architecture("t5_metro", "t5_metro_base_rpe")
def t5_metro_base_rpe(args):
    args.no_generator = safe_getattr(args, "no_generator", False)
    args.generator_sample_mode = safe_getattr(args, "generator_sample_mode", "train")
    args.generator_layers = safe_getattr(args, "generator_layers", 3)
    return transformer_t5_base_rpe(args)


@register_model_architecture("t5_metro", "t5_metro_large_rpe")
def t5_metro_large_rpe(args):
    args.no_generator = safe_getattr(args, "no_generator", False)
    args.generator_sample_mode = safe_getattr(args, "generator_sample_mode", "train")
    args.generator_layers = safe_getattr(args, "generator_layers", 6)
    return transformer_t5_large_rpe(args)
