import argparse
from Model import train_and_save_model
from ImageUtils import check_positive_integral_value

# data_directory -> required directory from which to train
# --save_dir save_directory -> optional directory to save checkpoint
# --arch "vgg11" -> optional architecture for features
# --learning_rate 0.003 -> optional learning rate
# --hidden_units 512 -> optional number of hidden units in classifier
# --epochs 5 -> optional number of epochs in training
# --gpu -> optional choice to use GPU during training

# prints out training loss, validation loss, validation accuracy as network trains

# I assume that data_directory will contain sub-directories '/train', '/valid', '/test'

def getTrainingDataArguments():
    parser = argparse.ArgumentParser(description="Train a model based on input data.")
    parser.add_argument('data_directory', help="The directory containing the three sub-directories for training, \
                        validation, and testing.")
    parser.add_argument('--save_dir', dest='save_directory', default='command_line_checkpoint', help="The directory into \
                        which to save the checkpoint.")
    parser.add_argument('--arch', dest='architecture', default='vgg11', help="The architecture type to use for getting \
                        a feature set during transfer learning.")
    parser.add_argument('--learning_rate', type=float, default=0.003, help="The learning rate of the neural network, \
                        i.e. the step size.")
    parser.add_argument('--hidden_units', type=check_positive_integral_value, default=256, help="The number of hidden \
                        units in the classifier used following the features in the transfer learning process.")
    parser.add_argument('--epochs', type=check_positive_integral_value, default=5, help="The number of times that the \
                        feed forward and backward pass algorithm should be run.")
    parser.add_argument('--gpu', dest='is_gpu_enabled', action='store_true', default=False, help="A boolean flag determining whether \
                        the algorithm should be run with the GPU; this is recommended when using large neural networks such as VGG.")
    return parser.parse_args()


if __name__ == "__main__":
    args = getTrainingDataArguments()
    train_and_save_model(args.data_directory, args.save_directory, args.architecture, args.learning_rate, args.hidden_units,
                         args.epochs, args.is_gpu_enabled)
