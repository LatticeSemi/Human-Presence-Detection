# Steps to follow to run augmentation on dataset.

# The script expects labels in kitty format as metioned in example below.
# EG : <Class_Name> 0.0 0.0 0.0 X1 Y1 X2 Y2

# Use below command to run augmentation on given input dataset.

python augmentation.py --image_dir <input_image_direcory_path> \
						--label_dir <input_label_direcory_path> \
						--out_image_dir <output_image_direcory_path> \
						--out_label_dir <output_label_direcory_path>

# The configurations for augmentation can be changed in config.py
# The operations for augmentation can be opted in list "all_op" located at line number 224 in augmentation.py.
