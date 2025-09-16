python run_attention_visualization.py \
    --model_path /data/VisualPrism/checkpoints/Tokenpacker-7b \
    --image_path /data/VisualPrism/OIP.jpg \
    --text "What's in the image?" \
    --output_dir ./attention_results_Tokenpacker \
    --layers 0 8 16 23 \
    --max_new_tokens 100