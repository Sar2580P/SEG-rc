#!/bin/bash

# Define arrays for different hyperparameters
seeds=(1828499611299255970 5867991739709137916 97 42)
blur_time_regions=("begin" "mid" "end" "begin mid end")
seg_applied_layers=("mid" "down" "up" "mid down" "mid up" "down up" "mid down up")
metric_tracked_block=("mid" "down" "up")

# Logging function
echo_log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Error handling function that continues execution
handle_error() {
    echo_log "An error occurred in this iteration, moving to next..."
}

# Iterate over all combinations of seed, blur_time_region, seg_applied_layer, and metric_tracked_block
for seed in "${seeds[@]}"; do
    for blur_time_region in "${blur_time_regions[@]}"; do
        for seg_applied_layer in "${seg_applied_layers[@]}"; do
            for metric_block in "${metric_tracked_block[@]}"; do
                config_file="config.yaml"

                echo_log "Updating config: $config_file"
                {
                    cat > "$config_file" <<EOL
save_attn_map: !!bool False
seed: $seed
num_inference_steps: 30
seg_applied_layers: [$(echo "$seg_applied_layer" | sed 's/ /, /g')]
blur_time_regions: [$(echo "$blur_time_region" | sed 's/ /, /g')]
metric_tracked_block: $(echo "$metric_block" | sed 's/ /, /g')
EOL
                } || handle_error

                echo_log "Running main.py with config: $config_file"
                {
                    python main.py
                } || handle_error

                echo_log "Completed run for $config_file"
            done
        done
    done
done

echo_log "All configurations processed successfully."
