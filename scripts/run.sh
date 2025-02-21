#!/bin/bash

# Define the variables
seeds=(77)     # 77  5867991739709137916
blur_time_regions=("begin")  #  "mid" "end" "begin mid end"
seg_applied_layers=("mid")   #   "down" "up" "down, mid, up"
metric_tracked_block=("mid")   #  "down" "up"
num_inference_steps=(30)    # 20 30 40
# gaussian_techniques=("gaussian_17_10" "gaussian_31_10" "gaussian_-1 _10" "gaussian_-1_1000")
interpolatedBoxBlur_techniques=("interpolatedBoxBlur_5_0.9" "interpolatedBoxBlur_17_0.9" "interpolatedBoxBlur_31_0.9" "interpolatedBoxBlur_61_0.9")   # "interpolatedBoxBlur_31_0.5" "interpolatedBoxBlur_31_0.6" "interpolatedBoxBlur_31_0.7" "interpolatedBoxBlur_31_0.8" "interpolatedBoxBlur_31_0.99"
temperatureAnnealing_techniques=("temperatureAnnealing_linear_2" "temperatureAnnealing_linear_2.4" "temperatureAnnealing_cosine_4" "temperatureAnnealing_cosine_6" "temperatureAnnealing_exponential_2.5" "temperatureAnnealing_exponential_3.3")
gaussian_techniques=("gaussian_3_10" "gaussian_7_10" "gaussian_17_10" "gaussian_-1_10" "gaussian_-1_10000" "gaussian_31_10")
ema_techniques=("ema_0.85_0.9_linear" "ema_0.95_0.99_linear" "ema_0.50_0.65_linear" "ema_0.6_0.75_linear" "ema_0.75_0.99_linear")
# blurring_technique=("${interpolatedBoxBlur_techniques[@]}")
blurring_technique=("${gaussian_techniques[@]}" "${ema_techniques[@]}" "${interpolatedBoxBlur_techniques[@]}")
guidance_scale=(0 5)    #
seg_scale=(0 3)   # 0

# Loop over all combinations
for seed in "${seeds[@]}"; do
    for blur in "${blur_time_regions[@]}"; do
        for seg_layer in "${seg_applied_layers[@]}"; do
            for metric_block in "${metric_tracked_block[@]}"; do
                for steps in "${num_inference_steps[@]}"; do
                    for blur_tech in "${blurring_technique[@]}"; do
                        for guide in "${guidance_scale[@]}"; do
                            for seg in "${seg_scale[@]}"; do

                                # Update config.yaml with sed
                                sed -i "s/^seed:.*/seed: $seed/" config.yaml
                                sed -i "s/^num_inference_steps:.*/num_inference_steps: $steps/" config.yaml
                                sed -i "s/^seg_applied_layers:.*/seg_applied_layers: [$seg_layer]/" config.yaml
                                sed -i "s/^blur_time_regions:.*/blur_time_regions: [$blur]/" config.yaml
                                sed -i "s/^metric_tracked_block:.*/metric_tracked_block: $metric_block/" config.yaml
                                sed -i "s|^blurring_technique:.*|blurring_technique: !!str '$blur_tech'|" config.yaml
                                sed -i "s/^guidance_scale :.*/guidance_scale : $guide/" config.yaml
                                sed -i "s/^seg_scale :.*/seg_scale : $seg/" config.yaml

                                # Run Python script
                                python main.py
                            done
                        done
                    done
                done
            done
        done
    done
done
