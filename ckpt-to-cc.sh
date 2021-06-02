#========================================================================
#========================================================================
# Parameters
#========================================================================

#========================================================================
# 64x64x1 input case
#========================================================================
TRAIN_DIR=train
CHKPOINT_DIR=./logs/humandet/$TRAIN_DIR
CHKPOINT_FILE=model.ckpt-249999
OUTPUT_SUB_DIR=$TRAIN_DIR

EVAL_PBTXT_FILE=model.pbtxt # pbtxt made in eval/demo mode, not training mode
INPUT_NODE_NAMES=batch
INPUT_SHAPE=1,64,64,1

OUTPUT_NODE_NAMES=conv12/convolution #Reshape_1 <-- This is what we use!!!

#========================================================================
OUTPUT_DIR=./output/$OUTPUT_SUB_DIR
OUTPUT_FROZEN_PB_NAME=humancnt_frozen.pb  # frozen .pb
OUTPUT_TFLITE_NAME=humancnt_frozen.tflite # TF Lite
OUTPUT_C_NAME=humancnt_frozen.cc          # C++ code
#========================================================================
FREEZE_TOOL=CKPT2PB
CONVERSION_TOOL=gen_tflite_and_quant.py #tflite_convert # toco
# TFLITE_OPTIMIZE=True
# TFLITE_QUANT=FullInt8 # Dynamic, FullInt8, Float16

# Dynamic  - 4x smaller, 2x-3x speed up; for CPU (mix of integer and float32); blob uses float32 size
# FullInt8 - 4x smaller, 3x+ speed up; for CPU, TPU, MCU; accuracy goes down... <--- What we use
# Float16  - 2x smaller; for GPU; not supported in TFLiteMCU...
#========================================================================
#========================================================================
#========================================================================
# Frozen .pb generation from .meta, checkpoint and pbtxt
#
# We have two tools:
# 1) Base tool: git clone https://github.com/tensorflow/tensorflow
#    This base tool is known to cause float_ref issue when reading to make
#    tflite (open discussion). So, we modified it (tensorflow/tensorflow/
#    python/tools/freeze_graph.py) to add fixes (as we implmented it in ckpt2pb.py)
#
#    The frozen .pb has original dimension (64*128*128*1) and it causes issue
#    with TFLite converter that expects BatchSize=1
#
# 2) ckpt2pb: Built based on stackoverflow. Handles .meta, checkpoint, and pbtxt
#    and generates a fronzen .pb. The generated .pb has no dimension except
#    the # of channels (e.g., ?*?*?*16)
# 
# We use #2. 
#
# pbtxt has to come from demo/eval with training = FALSE. Otherwise BN does not work
# The fix of manual network after getting pbtxt from traing mode does not work.
# IF we get pbtxt from demo (traing = FALSE), there is no such bugs in the pbtxt!
#========================================================================

export CUDA_VISIBLE_DEVICES=-1

python src/ckpt2pb.py \
	--check_point_dir=$CHKPOINT_DIR \
	--check_point_file=$CHKPOINT_FILE \
	--eval_pbtxt_file=$EVAL_PBTXT_FILE \
	--output_node_names=$OUTPUT_NODE_NAMES \
	--output_dir=$OUTPUT_DIR \
	--output_file=$OUTPUT_FROZEN_PB_NAME


#========================================================================
# Converting to TF lite format
# - tflite_convert from TF 1.9. For previous versions, toco
# - for quantization, tflite_quant_conv.py is used 
#========================================================================
python src/$CONVERSION_TOOL \
	--output_file=$OUTPUT_DIR/$OUTPUT_TFLITE_NAME \
	--graph_def_file=$OUTPUT_DIR/$OUTPUT_FROZEN_PB_NAME \
	--input_arrays=$INPUT_NODE_NAMES \
	--input_shape=$INPUT_SHAPE \
	--output_arrays=$OUTPUT_NODE_NAMES \
	--input_path=./data/input_images/faces \
	#--optimization=$TFLITE_OPTIMIZE \
	#--quantization=$TFLITE_QUANT


#========================================================================
# Converting TF lite format to C code
#========================================================================
xxd -i $OUTPUT_DIR/$OUTPUT_TFLITE_NAME > $OUTPUT_DIR/$OUTPUT_C_NAME
