#! /bin/bash
export PYTHONPATH=$HOME/dygiepp:$PYTHONPATH

# 설정
INPUT_FILE="/home/sua/dygiepp/dygie_input_scierc_sampling.jsonl"       # 원본 input
CHUNK_DIR="/home/sua/dygiepp/chunks_input"                             # chunk 저장
OUTPUT_DIR="/home/sua/dygiepp/chunks_output"                           # chunk 별 predict 결과
FINAL_OUTPUT="/home/sua/dygiepp/scierc_output.jsonl"                   # 최종 merge 결과
CHUNK_SIZE=5                                                           # chunk 당 문서 수
MODEL_PATH="/home/sua/dygiepp/pretrained_model/scierc.tar.gz"          # AllenNLP 모델 경로

mkdir -p $CHUNK_DIR
mkdir -p $OUTPUT_DIR


# 1. JSONL chunk 나누기
echo "[1/3] Splitting input into chunks..."
split -l $CHUNK_SIZE -d -a 3 --additional-suffix=.jsonl $INPUT_FILE $CHUNK_DIR/chunk_


# 2. chunk 별 predict
echo "[2/3] Running model prediction for each chunk..."
for CHUNK_FILE in $CHUNK_DIR/*.jsonl; do
    BASENAME=$(basename $CHUNK_FILE .jsonl)
    OUT_FILE="$OUTPUT_DIR/${BASENAME}_output.jsonl"

    echo "Processing $CHUNK_FILE -> $OUT_FILE"

    allennlp predict $MODEL_PATH $CHUNK_FILE \
        --output-file $OUT_FILE \
        --silent \
        --predictor dygie \
        --cuda-device -1 \
        --include-package dygie \
        --batch-size 1 \
        --use-dataset-reader \
        --overrides '{"model":{"embedder":{"token_embedders":{"bert":{"model_name":"/home/sua/dygiepp/bert_models/scibert_scivocab_cased"}}}}}'
done


# 3. merge chunk 결과
echo "[3/3] Merging all outputs..."
cat $OUTPUT_DIR/*_output.jsonl > $FINAL_OUTPUT
echo "Final output: $FINAL_OUTPUT"

echo "DONE!"
