#!/bin/bash

python -m llava.eval.model_vqa \
    --model-path llava-visualprism-7b \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/llava-visualprism-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/llava-visualprism-7b.jsonl \
    --dst ./playground/data/eval/mm-vet/results/llava-visualprism-7b.json
