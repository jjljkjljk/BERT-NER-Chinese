python train_my_model.py \
    --device gpu \
    --vocab_path config/vocab.txt \
    --config_path config/config.json \
    --output_path output/bert_softmax_mine \
    --lr 5e-5 \
    --eps 1.0e-08 \
    --epochs 100 \
    --batch_size_train 4 \
    --batch_size_eval 4 \
    --num_workers 0 \
    --eval_step 25 \
    --max_len 150 \
    --data_path datasets/cner/ \
    --dataset_name cner \
    --seed 42 \
    --markup bios \
    --grad_acc_step 1 \
    --max_grad_norm 1.0 \
    --num_workers 0 \
    --warmup_proportion 0.1 \
    --do_train \
    --do_eval