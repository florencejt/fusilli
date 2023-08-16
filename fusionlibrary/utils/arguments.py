import argparse
import sys


def init_parser():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Run fusion models")
    # Add the argument for prediction type
    parser.add_argument(
        "-p",
        "--pred_type",
        choices=["regression", "binary", "multiclass"],
        help="Type of prediction",
        required=True,
    )
    # Add the boolean argument for running on the cluster
    parser.add_argument(
        "-c",
        "--cluster",
        action="store_true",
        help="Run on the cluster (default false)",
        default=False,
    )

    # Add the boolean argument for kfold training and the number of cross validation folds
    parser.add_argument(
        "-k", "--kfold", action="store_true", help="Enable kfold training"
    )
    parser.add_argument(
        "--num_folds",
        type=int,
        help="Number of cross validation folds",
        required="--kfold" in sys.argv or "-k" in sys.argv,
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        help="Number of classes",
        default=3,
        required="--pred_type" in sys.argv and "multiclass" in sys.argv,
    )

    # Add the boolean argument for logging with wandb
    parser.add_argument(
        "-l",
        "--log",
        action="store_true",
        default=False,
        help="Enable logging with wandb",
    )

    # Add the argument for the number of repetitions
    parser.add_argument(
        "-n", "--num_reps", default=1, type=int, help="Number of repetitions"
    )

    return parser
