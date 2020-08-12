import argparse
import sys
import os
from .config import EXAMPLES_PATH,  MODEL_DIR_PATH
from .live_predictions import LivePredictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=[
                        'live'], help='The action is to make live predictions.')

    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument(
        "-p", "--path", help="location of the audio file.", required=True)

    args = parser.parse_args()
    if args.action == 'live':
        input_path = args.path
        file = EXAMPLES_PATH + input_path

        if not os.path.isfile(file):
            print('The file specified does not exist', file)
            sys.exit()

        # live_prediction = LivePredictions(file=EXAMPLES_PATH + '03-01-01-01-01-02-05.wav')
        live_prediction = LivePredictions(file=file)
        live_prediction.loaded_model.summary()
        print("Prediction is: ", live_prediction.make_predictions())
        # live_prediction = LivePredictions(file=EXAMPLES_PATH + '10-16-07-29-82-30-63.wav')
        # live_prediction.make_predictions()


if __name__ == '__main__':
    main()
