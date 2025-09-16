python generate_attention_heatmap.py \
    --model_path /data/VisualPrism/checkpoints/Tokenpacker-7b \
    --image_path /data/VisualPrism/OIP.jpg \
    --text "What's in the image?" \
    --output_dir ./attention_heatmaps_Tokenpacker \
    --layers 0 8 16 20 23 \
    --max_new_tokens 100