import argparse
import functools
import json
import os
import sys
from dataclasses import dataclass
from itertools import chain
from typing import Optional, Union

import numpy as np
import torch
from datasets import load_dataset, load_metric
from promptsource.templates import TemplateCollection
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy

TASK_LENS = [
    ('high_school_biology', 32),
    ('jurisprudence', 11),
    ('prehistory', 35),
    ('high_school_microeconomics', 26),
    ('nutrition', 33),
    ('high_school_geography', 22),
    ('human_sexuality', 12),
    ('astronomy', 16),
    ('moral_scenarios', 100),
    ('clinical_knowledge', 29),
    ('electrical_engineering', 16),
    ('econometrics', 12),
    ('high_school_computer_science', 9),
    ('college_biology', 16),
    ('miscellaneous', 86),
    ('high_school_mathematics', 29),
    ('college_medicine', 22),
    ('high_school_macroeconomics', 43),
    ('us_foreign_policy', 11),
    ('professional_law', 170),
    ('high_school_government_and_politics', 21),
    ('security_studies', 27),
    ('public_relations', 12),
    ('global_facts', 10),
    ('marketing', 25),
    ('high_school_chemistry', 22),
    ('machine_learning', 11),
    ('sociology', 22),
    ('moral_disputes', 38),
    ('college_physics', 11),
    ('high_school_statistics', 23),
    ('management', 11),
    ('virology', 18),
    ('high_school_physics', 17),
    ('high_school_world_history', 26),
    ('international_law', 13),
    ('logical_fallacies', 18),
    ('world_religions', 19),
    ('professional_accounting', 31),
    ('elementary_mathematics', 41),
    ('conceptual_physics', 26),
    ('college_computer_science', 11),
    ('human_aging', 23),
    ('high_school_psychology', 60),
    ('college_mathematics', 11),
    ('medical_genetics', 11),
    ('abstract_algebra', 11),
    ('professional_medicine', 31),
    ('computer_security', 11),
    ('philosophy', 34),
    ('business_ethics', 11),
    ('professional_psychology', 69),
    ('high_school_us_history', 22),
    ('high_school_european_history', 18),
    ('college_chemistry', 8),
    ('formal_logic', 14),
    ('anatomy', 14)
]


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
            sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
            maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
            different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
            Note that it's very NOT recommended to use fp16 to do any time of inference with T0 as the predictions will vastly differ from the predictions using fp32.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [
                {
                    k: v[i]
                    for k, v in feature.items()
                    if k != "targets"
                }
                for i in range(num_choices)
            ]
            for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # Pad the labels because it's not padded automatically
        max_label_length = max([len(elem["labels"]) for elem in flattened_features])
        batch["labels"] = [
            l + [self.tokenizer.pad_token_id] * (max_label_length - len(l))
            for l in [elem["labels"] for elem in flattened_features]
        ]
        batch["labels_attention_mask"] = [
            m + [0] * (max_label_length - len(m))
            for m in [elem["labels_attention_mask"] for elem in flattened_features]
        ]

        # Convert to tensors
        batch = {
            k: torch.tensor(v)
            for k, v in batch.items()
        }

        batch["targets"] = torch.tensor([f.pop("targets") for f in features])
        return batch


def _strip_bos_batch(items, bos_id):
    if bos_id is None:
        return items
    assert set(items.keys()) == {'input_ids', 'attention_mask'}
    r = {'input_ids': [], 'attention_mask': []}
    for seq in items['input_ids']:
        assert seq[0] == bos_id
        r['input_ids'].append(seq[1:])
    for seq in items['attention_mask']:
        r['attention_mask'].append(seq[1:])
    return r


def preprocess(tokenizer, template, column_names, padding, max_src_len, max_tgt_len, retrieval_aug, examples):
    bs = len(examples[column_names[0]])

    input_texts = []
    target_texts = []
    answer_choices_texts = []
    for i in range(bs):
        ex = {
            k: examples[k][i]
            for k in column_names
        }
        input, target = template.apply(ex)

        if retrieval_aug:
            input += "\n \n"
            for j in range(10):
                input += f"Reference passage {j + 1}: " + ex['passages'][j]

        ex_answer_choices = template.get_answer_choices_list(ex)
        assert target in ex_answer_choices
        input_texts.append(input)
        target_texts.append(target)
        answer_choices_texts.append(ex_answer_choices)

    tokenized_inputs = tokenizer(
        input_texts,
        padding=padding,
        max_length=max_src_len,
        truncation=True,
        add_special_tokens=True,
    )
    tokenized_targets = [
        _strip_bos_batch(
            tokenizer(
                ans_choi,
                padding=True,
                max_length=max_tgt_len,
                truncation=True,
            ),
            tokenizer.bos_token_id
        )
        for ans_choi in answer_choices_texts
    ]

    features = {
        k: [
            [elem for _ in range(len(tokenized_targets[idx]["input_ids"]))]
            for idx, elem in enumerate(v)
        ]
        for k, v in tokenized_inputs.items()
    }

    features["labels"] = [
        tokenized_targets[idx]["input_ids"]
        for idx in range(bs)
    ]
    features["labels_attention_mask"] = [
        tokenized_targets[idx]["attention_mask"]
        for idx in range(bs)
    ]
    features["targets"] = [
        answer_choices_texts[idx].index(t)
        for idx, t in enumerate(target_texts)
    ]

    return features


def main():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("ckpt_path", type=str)
    parser.add_argument("--retrieval_aug", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    data_files = {"validation": os.path.join(args.data_dir, "validation.jsonl")}
    raw_datasets = load_dataset("json", data_files=data_files)

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)

    collection = TemplateCollection()
    templates = collection.get_dataset("ai2_arc", "ARC-Challenge")
    template = templates["heres_a_problem"]

    column_names = raw_datasets["validation"].column_names
    padding = False
    max_src_len = 512
    max_tgt_len = 256

    preprocess_fn = functools.partial(
        preprocess,
        tokenizer, template, column_names, padding, max_src_len, max_tgt_len, args.retrieval_aug
    )

    eval_dataset = raw_datasets['validation'].map(
        preprocess_fn, batched=True, remove_columns=column_names,
        load_from_cache_file=False, keep_in_memory=True  # disable caching
    )

    data_collator = DataCollatorForMultipleChoice(tokenizer, pad_to_multiple_of=8)

    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.batch_size)

    hf_model = AutoModelForSeq2SeqLM.from_pretrained(args.ckpt_path)
    hf_model.half()
    hf_model.cuda()
    hf_model.eval()

    metric = load_metric("accuracy")
    all_predictions, all_references = [], []
    for batch in eval_dataloader:
        device = torch.device("cuda")
        for k in batch:
            batch[k] = batch[k].to(device)
        model_inputs = {
            k: batch[k]
            for k in ["input_ids", "attention_mask", "labels"]
        }
        with torch.no_grad():
            logits = hf_model(**model_inputs).logits
            masked_log_probs = batch["labels_attention_mask"].unsqueeze(-1) * torch.log_softmax(logits, dim=-1)
            seq_token_log_probs = torch.gather(masked_log_probs, -1, batch["labels"].unsqueeze(-1))
            seq_log_prob = seq_token_log_probs.squeeze(dim=-1).sum(dim=-1)
            seq_log_prob = seq_log_prob.view(batch["targets"].size(0), -1)
            predictions = seq_log_prob.argmax(dim=-1)
        metric.add_batch(predictions=predictions.cpu(), references=batch['targets'].cpu())
        all_predictions.append(predictions.cpu().numpy())
        all_references.append(batch['targets'].cpu().numpy())

    all_predictions = np.concatenate(all_predictions)
    all_references = np.concatenate(all_references)

    begin, end = 0, 0
    accs = []
    for task_name, task_len in TASK_LENS:
        begin, end = end, end + task_len
        acc = (all_predictions[begin: end] == all_references[begin: end]).sum() / task_len
        accs.append((task_name, acc))

    json.dump({
        "micro_accuracy": metric.compute(),
        "accuracy": np.mean([acc for task_name, acc in accs]),
        "accuracy_of_subtasks": accs
    }, sys.stdout, indent=4)


if __name__ == '__main__':
    main()
