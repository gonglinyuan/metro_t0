import argparse
import csv
import functools
import json
import multiprocessing as mp
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import datasets
import promptsource.utils
import torch
from filelock import FileLock
from omegaconf import DictConfig
from promptsource import templates

from fairseq.data import encoders

MAX_EXAMPLES_PER_DATASET = 500000

dataset_registry = {}

t0_eval = {
    "BASE": [],
    "BIAS_FAIRNESS": []
}
t0_train = {
    "BASE": [],
    # GPT3 evaluation set
    "GPT_EVAL": [],
    # SuperGLUE (except RTE and CB)
    "SGLUE": []
}

TASK_BLACKLIST = [
    # Tasks which often tokenize to > 1024 tokens currently
    "hotpot_qa_distractor_Generate_Explanations",
    "hotpot_qa_fullwiki_Generate_Explanations",
    "hotpot_qa_distractor_Generate_Answer_and_Explanations",
    "hotpot_qa_fullwiki_Generate_Answer_and_Explanations",
    "hotpot_qa_fullwiki_Generate_Answer",
    "hotpot_qa_distractor_Generate_Answer",
    "hotpot_qa_distractor_Generate_Title_2",
    "hotpot_qa_fullwiki_Generate_Title_2",
    "hotpot_qa_fullwiki_Generate_Title_1",
    "hotpot_qa_distractor_Generate_Title_1",
    "hotpot_qa_distractor_Generate_Question",
    "hotpot_qa_fullwiki_Generate_Question",
    "tab_fact_tab_fact_tab_fact_3",
    "tab_fact_tab_fact_tab_fact_2",
    "tab_fact_tab_fact_tab_fact_1",
    "tab_fact_tab_fact_tab_fact_7",
    "tab_fact_tab_fact_tab_fact_4",
    "tab_fact_tab_fact_tab_fact_5",
    "tab_fact_tab_fact_tab_fact_6",
    "wiki_hop_masked_Choose_Best_Object_Candidate",
    "wiki_hop_masked_Indirect_Question_about_Birthplace_Citizenship_Place_of_Death",
    "narrativeqa_Template_05",
    "ecthr_cases_alleged_violation_prediction_silver_rationales",
    # Tasks with broken cached files
    "gigaword_summarize_",
]

# Tasks that failed caching (won't try to fix them for now) - remove when we are done
D4_TRAIN_SCORE_EVAL_TASK_BLACKLIST = [
    "amazon_polarity_Is_this_product_review_positive",
    "amazon_polarity_Is_this_review_negative",
    "amazon_polarity_Is_this_review",
    "amazon_polarity_User_recommend_this_product",
    "amazon_polarity_convey_negative_or_positive_sentiment",
    "amazon_polarity_flattering_or_not",
    "amazon_polarity_negative_or_positive_tone",
    "amazon_polarity_user_satisfied",
    "amazon_polarity_would_you_buy",
    "dbpedia_14_given_a_choice_of_categories_",
    "dbpedia_14_given_list_what_category_does_the_paragraph_belong_to",
    "dbpedia_14_pick_one_category_for_the_following_text",
    "wiki_hop_original_choose_best_object_affirmative_1",
    "wiki_hop_original_choose_best_object_affirmative_2",
    "wiki_hop_original_choose_best_object_affirmative_3",
    "wiki_hop_original_choose_best_object_interrogative_1",
    "wiki_hop_original_choose_best_object_interrogative_2",
]

# Train tasks we don't care about evaluating on
D4_TRAIN_SKIP_EVAL = [
    "paws_labeled_final",
    "adversarial_qa_dbidaf",
    "adversarial_qa_dbert",
    "duorc_ParaphraseRC",
    "dream",
    "amazon_polarity",
    "app_reviews",
    "imdb",
    "wiki_bio",
    "gigaword",
    "multi_news",
    "samsum",
    "dbpedia_14",
    "trec",
]


def get_dataset_splits(dataset_name, subset_name=None):
    info = datasets.get_dataset_infos(dataset_name)
    subset_name = subset_name or list(info.keys())[0]
    return info[subset_name].splits


def task_clean(text):
    # Clean the text according to allowed characters for a task name
    return re.sub(r"[^\w\d\._]+", "_", text)


def get_task_name(dataset_name, subset_name, template_name):
    return task_clean(dataset_name + (f"_{subset_name}_" if subset_name is not None else "_") + template_name)


def try_apply_template(template, ex):
    ex = promptsource.utils.removeHyphen(ex)
    inputs_and_targets = template.apply(ex)
    return inputs_and_targets


def apply_template(template, ex):
    try:
        inputs_and_targets = try_apply_template(template, ex)
    except IndexError:  # answer choices list is empty
        return None
    if len(inputs_and_targets) == 2:
        inputs, targets = inputs_and_targets
        if targets == "":
            return None
        else:
            return inputs, targets
    else:
        return None


def whitespace_fix(s):
    return ' '.join(s.strip().split())


def process_example(tokenizer, bpe, ex):
    src = whitespace_fix(ex[0])
    tgt = whitespace_fix(ex[1])
    src = bpe.encode(tokenizer.encode(src)) + "\n"
    tgt = bpe.encode(tokenizer.encode(tgt)) + "\n"
    return src, tgt


def add_task(dataset_name, subset_name, template_name, task_name=None, split_mapping=None):
    task_name = task_name or get_task_name(dataset_name, subset_name, template_name)
    assert task_name not in dataset_registry
    dataset_splits = get_dataset_splits(dataset_name, subset_name)
    split_mapping = split_mapping or {k: k for k in dataset_splits.keys()}
    dataset_registry[task_name] = (dataset_name, subset_name, template_name, split_mapping)


def process_task(task_name, tokenizer, bpe, n_workers, out_dir, mode):
    def _is_split_good(splt):
        if mode == "train":
            return splt.startswith("train")
        elif mode == "valid":
            return not splt.startswith("train")
        else:
            raise ValueError(mode)

    dataset_name, subset_name, template_name, split_mapping = dataset_registry[task_name]
    template = all_templates.get_dataset(dataset_name, subset_name)[template_name]

    os.makedirs(os.path.join(out_dir, "examples"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, f"tokenized_{mode}"), exist_ok=True)

    apply_template_fn = functools.partial(apply_template, template)
    all_examples = {}
    for split in split_mapping.keys():
        if not _is_split_good(split):
            continue
        with FileLock(os.path.join(out_dir, "examples", f"{task_name}.{split}.lock")):
            if not os.path.exists(os.path.join(out_dir, "examples", f"{task_name}.{split}.pt")):
                raw_dataset = datasets.load_dataset(dataset_name, subset_name, split=split_mapping[split])
                print("Applying templates to", task_name, split)
                print("Example:", try_apply_template(template, raw_dataset[0]))
                with mp.Pool(n_workers) as pool:
                    examples = pool.map(apply_template_fn, raw_dataset)
                torch.save(examples, os.path.join(out_dir, "examples", f"{task_name}.{split}.pt"))
            examples = torch.load(os.path.join(out_dir, "examples", f"{task_name}.{split}.pt"))
        n_examples_before = len(examples)
        examples = list(filter(lambda it: it is not None, examples))
        n_examples_after = len(examples)
        if n_examples_before != n_examples_after:
            print(task_name, split, n_examples_before - n_examples_after, "examples are None")
        all_examples[split] = examples

    process_example_fn = functools.partial(process_example, tokenizer, bpe)
    with FileLock(os.path.join(out_dir, f"{task_name}.tokenization.lock")):
        if not (
            os.path.exists(os.path.join(out_dir, f"tokenized_{mode}", f"{task_name}.bpe.src"))
            and os.path.exists(os.path.join(out_dir, f"tokenized_{mode}", f"{task_name}.bpe.tgt"))
        ):
            f_src = open(os.path.join(out_dir, f"tokenized_{mode}", f"{task_name}.bpe.src"), "w", encoding="utf-8")
            f_tgt = open(os.path.join(out_dir, f"tokenized_{mode}", f"{task_name}.bpe.tgt"), "w", encoding="utf-8")
            for split, split_examples in all_examples.items():
                if not _is_split_good(split):
                    continue
                with mp.Pool(n_workers) as pool:
                    for src, tgt in pool.map(process_example_fn, split_examples):
                        f_src.write(src)
                        f_tgt.write(tgt)
            f_src.close()
            f_tgt.close()
        with open(os.path.join(out_dir, f"tokenized_{mode}", f"{task_name}.bpe.src"), "r", encoding="utf-8") as f_src:
            n_src = sum(1 for _ in f_src)
        with open(os.path.join(out_dir, f"tokenized_{mode}", f"{task_name}.bpe.tgt"), "r", encoding="utf-8") as f_tgt:
            n_tgt = sum(1 for _ in f_tgt)
    n_total = 0
    for split, split_examples in all_examples.items():
        if not _is_split_good(split):
            continue
        n_total += len(split_examples)
    assert n_src == n_tgt == n_total > 0, f"{n_src}  {n_tgt}  {n_total}"
    return n_total


def collect_data_info():
    with open("t0_datasets.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["subset"] == "":
                row["subset"] = None  # to match promptsource.Template object
            dataset_subset = (row["HF_name"], row["subset"])
            if row["do_train"] != "":
                do_train_source = row["do_train"]
                # sanity checks
                if do_train_source == "SGLUE":
                    assert dataset_subset[0] == "super_glue"
                t0_train[do_train_source].append(dataset_subset)
            if row["do_eval"] != "":
                do_eval_source = row["do_eval"]
                # sanity checks
                if do_eval_source == "BIAS_FAIRNESS":
                    assert row["task_by_convention"] == "bias_and_fairness"
                t0_eval[do_eval_source].append(dataset_subset)

    all_datasets = sum(t0_train.values(), []) + sum(t0_eval.values(), [])
    all_templates = templates.TemplateCollection()
    all_templates.remove("anli")  # Need to special-case ANLI due to weird split conventions

    # 3 stages of training/ablation: D4 -> GPT -> SuperGLUE
    t0_train_mixture: Dict[str, List[str]] = {key: [] for key in t0_train}
    t0_eval_mixture: Dict[str, List[str]] = {key: [] for key in t0_eval}
    single_original_task: Dict[Tuple[str, str], str] = {}
    all_original_tasks: List[str] = []
    for dataset_name, subset_name in list(all_templates.keys):
        if (dataset_name, subset_name) not in all_datasets:
            all_templates.remove(dataset_name, subset_name)
            continue
        dataset = all_templates.get_dataset(dataset_name, subset_name)
        for template_name in dataset.all_template_names:
            add_task(dataset_name, subset_name, template_name)
            template = dataset[template_name]
            task_name = get_task_name(dataset_name, subset_name, template_name)

            if (dataset_name, subset_name) not in single_original_task and template.metadata.original_task:
                single_original_task[(dataset_name, subset_name)] = task_name

            if template.metadata.original_task:
                all_original_tasks.append(task_name)

            # Check that the dataset_subset_tuple is in t0_train
            for key, dataset_subset_tuples in t0_train.items():
                if (dataset_name, subset_name) in dataset_subset_tuples:
                    t0_train_mixture[key].append(task_name)

            # Check that the dataset_subset_tuple is in t0_eval
            if (dataset_name, subset_name) in t0_eval["BASE"]:
                if template.metadata.original_task:
                    t0_eval_mixture["BASE"].append(task_name)
                # TODO use template.metadata.answer_choices here for rank eval
            if (dataset_name, subset_name) in t0_eval["BIAS_FAIRNESS"]:
                t0_eval_mixture["BIAS_FAIRNESS"].append(task_name)

    # Special case for ANLI, which has weirdly-named splits and rounds that should be subsets
    dataset_name, subset_name = ("anli", None)
    dataset = all_templates.get_dataset(dataset_name, subset_name)
    for anli_round in ("r1", "r2", "r3"):
        for template_name in all_templates.get_dataset(dataset_name, subset_name).all_template_names:
            task_name = get_task_name(dataset_name, subset_name, template_name) + f"_{anli_round}"
            split_mapping = {
                "train": f"train_{anli_round}",
                "validation": f"dev_{anli_round}",
                "test": f"test_{anli_round}",
            }
            add_task(dataset_name, subset_name, template_name, task_name, split_mapping)

            template = dataset[template_name]
            if template.metadata.original_task:
                t0_eval_mixture["BASE"].append(task_name)  # TODO or add to ANLI special mixture
            # TODO use template.metadata.answer_choices here for rank eval

    return all_templates, t0_train_mixture, all_original_tasks


all_templates, t0_train_mixture, all_original_tasks = collect_data_info()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mixture_name", type=str)
    parser.add_argument("--sp_path", type=str)
    parser.add_argument("--lower", action="store_true")
    parser.add_argument("--n_workers", type=int)
    parser.add_argument("--out_dir", type=str)
    args = parser.parse_args()

    tokenizer = encoders.build_tokenizer(
        DictConfig(dict(
            _name="guoke",
            lower=args.lower
        ))
    )
    bpe = encoders.build_bpe(
        DictConfig(dict(
            _name="sentencepiece",
            sentencepiece_model=args.sp_path,
        ))
    )

    if args.mixture_name == "t0_train":
        mixture = [
            task
            for task in t0_train_mixture["BASE"]
            if task not in TASK_BLACKLIST
        ]
        mode = "train"
    elif args.mixture_name == "t0p_train":
        mixture = [
            task
            for task in t0_train_mixture["BASE"] + t0_train_mixture["GPT_EVAL"]
            if task not in TASK_BLACKLIST
        ]
        mode = "train"
    elif args.mixture_name == "t0pp_train":
        mixture = [
            task
            for task in t0_train_mixture["BASE"] + t0_train_mixture["GPT_EVAL"] + t0_train_mixture["SGLUE"]
            if task not in TASK_BLACKLIST
        ]
        mode = "train"
    elif args.mixture_name == "t0_train_eval":
        mixture = [
            task
            for task in t0_train_mixture["BASE"]
            if (
                task not in TASK_BLACKLIST
                and task not in D4_TRAIN_SCORE_EVAL_TASK_BLACKLIST
                and not any([skip in task for skip in D4_TRAIN_SKIP_EVAL])
                and task in all_original_tasks
            )
        ]
        mode = "valid"
    else:
        raise ValueError()

    data_n_prompts = defaultdict(int)
    data_cap = {}
    for task in mixture:
        dataset_name, subset_name, _, _ = dataset_registry[task]
        data_n_prompts[(dataset_name, subset_name)] += 1
    for task in mixture:
        dataset_name, subset_name, _, _ = dataset_registry[task]
        data_cap[task] = MAX_EXAMPLES_PER_DATASET // data_n_prompts[(dataset_name, subset_name)]

    data_stats = {}

    for task in mixture:
        n_total = process_task(task, tokenizer, bpe, args.n_workers, args.out_dir, mode)
        data_stats[task] = n_total

    with open(os.path.join(args.out_dir, "DATASETS"), "w", encoding="utf-8") as f:
        for task in data_stats.keys():
            f.write(task + "\n")
    with open(os.path.join(args.out_dir, "data_cap.json"), "w", encoding="utf-8") as f:
        json.dump(data_cap, f, indent=4)
    if mode == "valid":
        valid_datasets = {}
        for task in mixture:
            dataset_name, subset_name, _, _ = dataset_registry[task]
            key = (
                dataset_name + "_" + subset_name
                if subset_name
                else dataset_name
            )
            if key not in valid_datasets:
                valid_datasets[key] = []
            valid_datasets[key].append(task)
        with open(os.path.join(args.out_dir, "valid_datasets.json"), "w", encoding="utf-8") as f:
            json.dump(valid_datasets, f, indent=4)


if __name__ == '__main__':
    main()
