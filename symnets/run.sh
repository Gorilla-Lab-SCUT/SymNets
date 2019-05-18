#!/bin/bash
python main.py --data_path_source /data/domain_adaptation/Office31/  --src amazon --epochs 200  --num_classes 31 --print_freq 1 --test_freq 1 \
            --data_path_source_t /data/domain_adaptation/Office31/ --src_t webcam  --lr 0.01 --gamma 0.1 --weight_decay 1e-4 --workers 4 \
            --data_path_target  /data/domain_adaptation/Office31/ --tar webcam --pretrained  --flag symnet  --log office31


