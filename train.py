import argparse
from pathlib import Path
from training_scripts.train_autoencoder import train_autoencoder
from training_scripts.train_vae import train_vae

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=False,
        help="Model to train: ae, vae or b-vae (default: ae)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="Batch size",
    )

    parser.add_argument(
        "--bottleneck-size",
        type=int,
        required=True,
        help="Number of dimensions of the bottleneck (latent space)",
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Name of the model",
    )

    parser.add_argument(
        "--dataset-folder",
        type=Path,
        required=True,
        help="main dataset folder, following Pytorch ImageFolder structure",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        required=False,
        help="learning rate",
        default=1e-4
    )

    parser.add_argument(
        "--beta",
        type=float,
        required=False,
        help="Beta value (default: 1)",
        default=1.0
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.model == "ae":
        train_autoencoder(
            epochs=args.epochs,
            batch_size=args.batch_size,
            bottleneck_size=args.bottleneck_size,
            output_file=args.output_file,
            dataset=args.dataset_folder,
        )
    elif args.model == "vae":
        train_vae(
            epochs=args.epochs,
            batch_size=args.batch_size,
            bottleneck_size=args.bottleneck_size,
            output_file=args.output_file,
            dataset=args.dataset_folder,
            learning_rate=args.learning_rate,
            beta=args.beta,
        )