GPUID=6
export CUDA_VISIBLE_DEVICES=${GPUID}
python src/finetune.py \
--data_dir=./data/webnlg \
--learning_rate=4e-5 \
--num_train_epochs 32 \
--task graph2text \
--model_name_or_path=../weights/T5-base/ \
--train_batch_size=4 \
--eval_batch_size=4 \
--early_stopping_patience 5 \
--gpus 1 \
--N=0 \
--output_dir=./outputs-t5/base/off/e32-lr4 \
--max_source_length=384 \
--max_target_length=384 \
--val_max_target_length=384 \
--test_max_target_length=384 \
--eval_max_gen_length=384 \
--do_predict \
--eval_beams 3