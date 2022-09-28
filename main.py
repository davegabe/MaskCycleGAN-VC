from zipfile import ZipFile
from pathlib import Path
import requests
import shutil
from tqdm.auto import tqdm
import os
from torch import hub

from mask_cyclegan_vc.test import MaskCycleGANVCTesting
from args.cycleGAN_test_arg_parser import CycleGANTestArgParser
from mask_cyclegan_vc.train import MaskCycleGANVCTraining
from args.cycleGAN_train_arg_parser import CycleGANTrainArgParser


def download_preprocessed_dataset(zipurl, dataset_path):
    """
        Download the preprocessed dataset and save into dataset_path

        Args:
        zipurl: URL of the preprocessed dataset
        dataset_path: Path to save the preprocessed dataset
    """
    print("Downloading the preprocessed dataset...")
    # make an HTTP request within a context manager
    with requests.get(zipurl, stream=True) as r:
        # check header to get content length, in bytes
        total_length = int(r.headers.get("Content-Length"))  # type: ignore
        # implement progress bar via tqdm
        with tqdm.wrapattr(r.raw, "read", total=total_length, desc="") as raw:
            # save the output to a file
            with open(f"{dataset_path}/vcc2018_preprocessed.zip", 'wb') as output:
                shutil.copyfileobj(raw, output)

    # unzip the downloaded file
    print("Unzipping the preprocessed dataset...")
    with ZipFile(f"{dataset_path}/vcc2018_preprocessed.zip", 'r') as zipObj:
        zipObj.extractall(dataset_path)

    print("Done.")


def main():
    local_home = Path(__file__).parent.resolve()
    os.environ['NUMBA_CACHE_DIR'] = f"{local_home}/tmp"
    hub.set_dir(f"{local_home}/cache")
    # REMOTE setup
    zipfile = "vcc2018_preprocessed_SOTA.zip"
    URL = f"https://github.com/davegabe/davegabe/releases/download/Resources/"

    # PATH setup
    dataset_path = f"{local_home}/dataset"
    model_path = f"{local_home}/model"
    print("Dataset path:", dataset_path)
    print("Model path:", model_path)

    # TRAINING setup
    epochs_per_save = 300  # Every n epochs, save the model
    num_epochs = 1800  # Max epochs to train
    conversions_to_test = [ # (source, target) permutations of VCC2SF3 (SF), VCC2SM3 (SM), VCC2TF1 (TF) and VCC2TM1 (TM)
        #("VCC2SM3", "VCC2TF1"),
        ("VCC2SM3", "VCC2TM1"),
        ("VCC2SF3", "VCC2TF1"),
        ("VCC2SF3", "VCC2TM1"),        
    ]

    # Download the preprocessed dataset and save into dataset_path
    download_preprocessed_dataset(f"{URL}/{zipfile}", dataset_path)

    for source, target in conversions_to_test:
        # Start training process
        model = f"{source}_to_{target}"
        training_args = [
            "--name", model,
            "--save_dir", model_path,
            "--preprocessed_data_dir", f"{dataset_path}/vcc2018_training",
            "--speaker_A_id", source,
            "--speaker_B_id", target,
            "--epochs_per_save", f"{epochs_per_save}",
            "--epochs_per_plot", f"{epochs_per_save}",
            "--num_epochs", f"{num_epochs}",
            "--decay_after", "1e4",
            "--sample_rate", "22050",
            "--num_frames", "64",
            "--max_mask_len", "25",
            "--batch_size", "1",
            "--continue_train"
        ]
        parser_train = CycleGANTrainArgParser()
        args_train = parser_train.parse_args(training_args)
        cycleGAN = MaskCycleGANVCTraining(args_train)
        cycleGAN.train()

        # Start evaluation process
        evaluation_args = [
            "--name", model,
            "--save_dir", model_path,
            "--preprocessed_data_dir", f"{dataset_path}/vcc2018_evaluation",
            "--speaker_A_id", source,
            "--speaker_B_id", target,
            "--ckpt_dir", f"{model_path}/{model}/ckpts",
            "--model_name", "generator_A2B"
        ]
        parser_test = CycleGANTestArgParser()
        args_test = parser_test.parse_args(evaluation_args)
        tester = MaskCycleGANVCTesting(args_test)
        tester.test()


if __name__ == "__main__":
    main()