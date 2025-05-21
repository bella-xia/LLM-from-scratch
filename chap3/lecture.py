import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=int, default=0)

    args = parser.parse_args()