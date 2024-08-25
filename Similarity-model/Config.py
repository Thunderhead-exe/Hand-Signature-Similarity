#class Config:

img_size = (170,242) # the size that will be used to resize (rescale) the signature
input_size = (150, 220) # the final size of the signature, obtained by croping the center of image
canvas_size = (840, 1360) # The size of a canvas where the signature will be centered on. Should be larger than the signature

model_name = 'convnet' # name of the model architecture to use: 'vit_base_patch32_224_in21k' 'tf_efficientnetv2_b0' 'resnext50_32x4d' 'tresnet_m'
target_col = "label" # column name for the target variable in the dataset
projection2d = True # use 2D projections
seed = 42 # random seed for reproducibility
