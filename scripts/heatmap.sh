python visualization/generate_attention_heatmap.py \
    --model_path ./checkpoints/VisualPrism-7b \
    --image_path /data/VisualPrism/OIP.jpg \
    --text "What's in the image?" \
    --output_dir ./attention_heatmaps_VisualPrism \
    --layers 0 16 20 23 \
    --max_new_tokens 100
