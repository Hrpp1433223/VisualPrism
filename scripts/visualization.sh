python visualization/run_attention_visualization.py \
    --model_path ./checkpoints/VisualPrism-7b \
    --image_path /data/VisualPrism/OIP.jpg \
    --text "What's in the image?" \
    --output_dir ./attention_results_VisualPrism \
    --layers 0 8 16 23 \
    --max_new_tokens 100
