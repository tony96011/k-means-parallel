#!/bin/bash
K=5
NUM=10000
INPUT_FILE="../input/input_$NUM.txt"
OUTPUT_DATAPOINTS="output_datapoints_$NUM"
OUTPUT_CENTROID="output_centroid_$NUM.txt"

srun -N1 -n1 --gres=gpu:1 ./main $K "$INPUT_FILE" "$OUTPUT_DATAPOINTS" "$OUTPUT_CENTROID"

if diff "../Sequential/$OUTPUT_DATAPOINTS" "$OUTPUT_DATAPOINTS" >/dev/null; then
    echo "Files are identical"
else
    echo "Files are different! Something is wrong."
fi

rm "$OUTPUT_DATAPOINTS"