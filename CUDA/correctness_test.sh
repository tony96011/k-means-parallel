#!/bin/bash
K=5
NUM=10000
INPUT_FILE="../input/input_$NUM.txt"
OUTPUT_DATAPOINTS="output_datapoints_$NUM"
OUTPUT_CENTROID="output_centroid_$NUM"

srun -N1 -n1 --gres=gpu:1 ./main $K "$INPUT_FILE" "$OUTPUT_DATAPOINTS" "$OUTPUT_CENTROID"

if diff "../Sequential/$OUTPUT_DATAPOINTS" "$OUTPUT_DATAPOINTS" >/dev/null; then
    echo -e "\e[32m😊 Correct\e[0m"  # Green smiley face emoji
else
    echo -e "\e[31m☹️ Wrong.\e[0m"  # Red sad face emoji
fi


rm "$OUTPUT_DATAPOINTS"