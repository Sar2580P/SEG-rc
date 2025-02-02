#!/bin/bash

# Define arrays for different hyperparameters
seeds=(4 7 21 24 42 64 90 121 184 256)
blur_time_regions=(['begin'] ['mid'] ['end'] ['begin' 'mid' 'end'])
seg_applied_layers=(['lower'] ['mid'] ['upper'] ['lower' 'mid'] ['lower' 'upper'] ['mid' 'upper'])

# Logging function
echo_log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Error handling
trap 'echo_log "An error occurred. Exiting..."; exit 1' ERR

# Iterate over all combinations
for seed in "${seeds[@]}"; do
    for blur_time_region in "${blur_time_regions[@]}"; do
        for seg_applied_layer in "${seg_applied_layers[@]}"; do
            config_file="config.yaml"
            
            echo_log "Updating config: $config_file"
            cat > "$config_file" <<EOL
save_attn_map: !!bool False
seed: $seed
num_inference_steps: 30
seg_applied_layers: ${seg_applied_layer[@]}
blur_time_regions: ${blur_time_region[@]}
EOL
            
            echo_log "Running main.py with config: $config_file"
            python main.py
            echo_log "Completed run for $config_file"
        done
    done
done

echo_log "All configurations processed successfully."
