#!/bin/bash

if [ ! -f run_squad.py ]; then
    echo "'run_squad.py' not found! Please run under 'ALBERT-TF2.0/'"
    exit -1
fi

export SQUAD_DIR=SQuAD
export SQUAD_VERSION=v1.1
export ALBERT_DIR=base_2
export OUTPUT_DIR=squad_out_${SQUAD_VERSION}

for i in 3 4 5 6 7 8
do
  python3 generate_question.py ${SQUAD_DIR}/dev-${SQUAD_VERSION}.json ${SQUAD_DIR}/new-dev-$i.json $i
  python3 run_squad.py \
    --mode=predict \
    --input_meta_data_path=${OUTPUT_DIR}/squad_${SQUAD_VERSION}_meta_data \
    --train_data_path=${OUTPUT_DIR}/squad_${SQUAD_VERSION}_train.tf_record \
    --predict_file=${SQUAD_DIR}/new-dev-$i.json \
    --albert_config_file=${ALBERT_DIR}/config.json \
    --init_checkpoint=${ALBERT_DIR}/tf2_model.h5 \
    --spm_model_file=${ALBERT_DIR}/vocab/30k-clean.model \
    --train_batch_size=28 \
    --predict_batch_size=28 \
    --learning_rate=1e-5 \
    --num_train_epochs=3 \
    --model_dir=${OUTPUT_DIR} \
    --strategy_type=mirror
  cp ${OUTPUT_DIR}/predictions.json ${OUTPUT_DIR}/predictions-$i.json
  echo -n "$i word question: " >> ${OUTPUT_DIR}/prediction-summary.txt
  python ${SQUAD_DIR}/evaluate-v1.1.py ${SQUAD_DIR}/new-dev-$i.json ${OUTPUT_DIR}/predictions-$i.json >> ${OUTPUT_DIR}/prediction-summary.txt
done
