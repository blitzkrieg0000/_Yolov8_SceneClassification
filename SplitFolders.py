import splitfolders

# Real
input_dir="dataset/VR/Real"
output_dir="dataset/VR01/Real"

# Virtual
input_dir="dataset/VR/Virtual"
output_dir="dataset/VR01/Virtual"

splitfolders.ratio(
    input_dir, # The location of dataset
    output=output_dir, # The output location
    seed=42, # The number of seed
    ratio=(.7, .2, .1), # The ratio of splited dataset
    group_prefix=None, # If your dataset contains more than one file like ".jpg", ".pdf", etc
    move=False # If you choose to move, turn this into True
)