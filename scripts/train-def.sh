GPUID=3
export CUDA_VISIBLE_DEVICES=${GPUID}
python webnlg/finetune.py \
--data_dir=./NWOrder/bart/webnlg/data/webnlg-re \
--learning_rate=3e-5 \
--num_train_epochs 2 \
--task def2text \
--model_name_or_path=./weights/bart/def-unk/ \
--train_batch_size=4 \
--eval_batch_size=4 \
--early_stopping_patience 5 \
--gpus 1 \
--pooling_type avg \
--use_unk \
--output_dir=./NWOrder/bart/outputs/def/unk/off3 \
--max_source_length=384 \
--max_target_length=384 \
--val_max_target_length=384 \
--test_max_target_length=384 \
--eval_max_gen_length=384 \
--do_train --do_predict \
--eval_beams 3