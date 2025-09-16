#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path llava-visualprism-7b \
    --question-file /data/VisualPrism/playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder /data/VisualPrism/playground/data/eval/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/llava-visualprism-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/llava-visualprism-7b.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/llava-visualprism-7b.json
