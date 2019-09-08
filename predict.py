import argparse
from ImageUtils import check_positive_integral_value

# path to image
# checkpoint

# --top_k -> top k most likely results
# --category_names -> mapping of categories to real names
# --gpu -> use gpu for inference

def getPredictionDataArguments():
    parser = argparse.ArgumentParser(description="Make predictions about classification of input images using the trained model.")
    parser.add_argument('path_to_image', help="Path to image to be classified.")
    parser.add_argument('checkpoint', help="Checkpoint containing saved model to be used to make the prediction.")
    parser.add_argument('--top_k', default=5, type=check_positive_integral_value, help="The number of best matching \
                        classes to display.")
    parser.add_argument('--category_names', help="A mapping of categories to real names")
    parser.add_argument('--gpu', dest='is_gpu_enabled', action='store_true', default=False, help="A boolean flag determining whether \
                        the model should be run with the GPU; this is recommended when the model contains large neural networks such as VGG.")
    return parser.parse_args()

if __name__ == "__main__":
    args = getPredictionDataArguments()
    print(args)
    print('args.path_to_image: ', args.path_to_image)
    print('args.checkpoint: ', args.checkpoint)
    print('args.top_k: ', args.top_k)
    print('args.category_names: ', args.category_names)
    print('args.is_gpu_enabled: ', args.is_gpu_enabled)
