OUT_DIR=$1
DICT_PATH=$2
SP_PATH=$3
LOWER=$4
N_WORKERS=$5

mkdir -p "${OUT_DIR}"

python collect_t0_train.py \
  t0pp_train \
  --sp_path "${SP_PATH}" \
  "${LOWER}" \
  --n_workers "${N_WORKERS}" \
  --out_dir "${OUT_DIR}"

rm "${OUT_DIR}"/*.lock
rm "${OUT_DIR}"/examples/*.lock

while IFS= read -r TASK; do
  fairseq-preprocess --workers "${N_WORKERS}" \
    --source-lang src \
    --target-lang tgt \
    --trainpref "${OUT_DIR}/tokenized_train/${TASK}.bpe" \
    --destdir "${OUT_DIR}/T0-bin/train/${TASK}" \
    --srcdict "${DICT_PATH}" \
    --tgtdict "${DICT_PATH}"
done <"${OUT_DIR}/DATASETS"

cp "${DICT_PATH}" "${OUT_DIR}/T0-bin/"
cp "${DICT_PATH}" "${OUT_DIR}/T0-bin/train/"
cp "${OUT_DIR}/data_cap.json" "${OUT_DIR}/T0-bin/train/"
