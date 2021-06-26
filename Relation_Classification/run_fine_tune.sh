export DATA_DIR=./data
export TRAIN_DATA=$1
export TEST_DATA=$2
export LABEL_FILE=$3
export MAX_LENGTH=122
export LEARNING_RATE=2e-5
export WEIGHT_DECAY=0.01
export MODEL_NAME=roberta-base
export NUM_EPOCHS=3
export SEED=2
export OUTPUT_DIR_NAME=$1
export CURRENT_DIR=${PWD}
export OUTPUT_DIR=${CURRENT_DIR}/output/${MODEL_NAME}/${OUTPUT_DIR_NAME}

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
--seed $SEED \
--do_eval \
--do_train \
#--evaluate_during_training \
#--no_cuda
