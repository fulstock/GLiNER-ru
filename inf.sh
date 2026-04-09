python gliner_inference.py \
	  --input /home/student1/data/collection3-gliner/test.json \
      --output ./logs/zeroshot/collection3_zeroshot.json \
      --metrics_output ./logs/zeroshot/collection3_zeroshot_metrics.json \
      --model urchade/gliner_multi-v2.1 \
      --threshold 0.5 \
      --measure_time \
      --timing_output ./logs/zeroshot/collection3_timing_zeroshot.json