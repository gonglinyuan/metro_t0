## MMLU Evaluation

With retrieval augmentation:

```bash
python run_mmlu.py \
openmatch-research/src/UDIT/data_hf/mmlu_msmarco_ra_FiD/cache \
google/flan-t5-base \
--retrieval_aug
```

Without retrieval augmentation:

```bash
python run_mmlu.py \
openmatch-research/src/UDIT/data_hf/mmlu_msmarco_ra_FiD/cache \
google/flan-t5-base \
```