GPUID=2
export CUDA_VISIBLE_DEVICES=${GPUID}
python src/finetune.py \
--data_dir=./data/webnlg-t5-re \
--learning_rate=3e-5 \
--num_train_epochs 50 \
--task def2text \
--model_name_or_path=../weights/t5/small/def-unk/ \
--train_batch_size=4 \
--eval_batch_size=4 \
--early_stopping_patience 5 \
--gpus 1 \
--use_unk \
--output_dir=./outputs-t5/small/def50/off1 \
--max_source_length=384 \
--max_target_length=384 \
--val_max_target_length=384 \
--test_max_target_length=384 \
--eval_max_gen_length=384 \
--do_predict \
--eval_beams 3