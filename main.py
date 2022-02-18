from curses import window
import cv2
import numpy as np
import argparse

from intrusion_detection import IntrusionDetection

from utils import imshow


def main():

    import random
    parser = argparse.ArgumentParser(
        description='Un programma per prendere {}'.format(
            random.choice(['28', '29', '30', '30L', '30'])))

    parser.add_argument('--input',
                        metavar='I',
                        type=str,
                        default='./videos/rilevamento-intrusioni-video1.mp4',
                        help='The input video. default: the provided demo')

    parser.add_argument('--output',
                        metavar='O',
                        type=str,
                        default='./output.txt',
                        help='The output text file. default: ./output.txt')

    parser.add_argument('--debug',
                        action='store_true',
                        help='Show additional information')

    args = vars(parser.parse_args())

    with open(args['output'], 'w') as f:
        id = IntrusionDetection(args['input'],
                                debug=args['debug'],
                                output_file=f)
        id.start()


if __name__ == '__main__':
    main()
