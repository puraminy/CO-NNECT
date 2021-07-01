export DATA_DIR=./data
export TRAIN_DATA=$1
export TEST_DATA=$2
export LABEL_FILE=$3
export MAX_LENGTH=142
export BATCH_SIZE=12
export LEARNING_RATE=2e-5
export WEIGHT_DECAY=0.01
#export MODEL_NAME=roberta-base
#export MODEL_NAME=bert-base-multilingual-cased
export MODEL_NAME=xlm-roberta-base
export NUM_EPOCHS=3
export SEED=2
export OUTPUT_DIR_NAME=$1
export CURRENT_DIR=${PWD}
export OUTPUT_DIR=/drive3/pouramini/connect/${MODEL_NAME}/${OUTPUT_DIR_NAME}

#if [ -d $OUTPUT_DIR ]; then # && [ -z "$4" ]; then
#     echo "${OUTPUT_DIR} already eixts! "
#     exit 1
#fi

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR
# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

python fine_tune_cnRel.py --data_dir $DATA_DIR \
--train_data $TRAIN_DATA \
--test_data $TEST_DATA \
--model_name_or_path $MODEL_NAME \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--learning_rate $LEARNING_RATE \
--weight_decay $WEIGHT_DECAY \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--per_gpu_eval_batch_size $BATCH_SIZE \
--seed $SEED \
--do_eval \
--do_train \
#--evaluate_during_training \
#--no_cuda
