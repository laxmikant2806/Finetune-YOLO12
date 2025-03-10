#!/usr/bin/env python

import os
import sys
import logging
import random
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import warnings
import requests
from ultralytics import YOLO

import mlflow
mlflow.set_tracking_uri("http://localhost:5000")

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class CONFIG:
    """Configuration class for model training and inference parameters."""
    DEBUG = True
    DISPLAY_IMAGES = False
    FRACTION = 0.05 if DEBUG else 1.0
    SEED = 88
    CLASSES = [
        "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
        "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle"
    ]
    NUM_CLASSES_TO_TRAIN = 10
    EPOCHS = 3 if DEBUG else 50
    BATCH_SIZE = 16
    BASE_MODEL = "yolov8s"
    MODELS_DIR = os.path.join(os.getcwd(), "models")
    BASE_MODEL_WEIGHTS = os.path.join(MODELS_DIR, f"{BASE_MODEL}.pt")
    EXP_NAME = f"PPE {BASE_MODEL} Fine Tuning with {EPOCHS} Epochs"
    OPTIMIZER = "auto"
    LR = 1e-3
    LR_FACTOR = 0.01
    WEIGHT_DECAY = 5e-4
    DROPOUT = 0.0
    PATIENCE = 20
    PROFILE = False
    LABEL_SMOOTHING = 0.0
    CUSTOM_DATASET_DIR = "data/"
    OUTPUT_DIR = os.getcwd()


class Finetuning:
    """Class for finetuning and exporting a YOLO model."""

    def __init__(self):
        """Initialize the Finetuning class with configuration."""
        self.CONFIG = CONFIG
        self.model = None
        self.yaml_path = None
        self.img_properties = None

    def create_directory(self, dir_path):
        """Create a directory if it doesn't exist.

        Args:
            dir_path (str): Path to the directory to create.
        """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logging.info(f"Created directory: {dir_path}")
        else:
            logging.info(f"Directory already exists: {dir_path}")

    def download_file(self, url, dest_path):
        """Download a file from a URL to a destination path.

        Args:
            url (str): URL of the file to download.
            dest_path (str): Destination path to save the file.
        """
        if os.path.exists(dest_path):
            logging.info(f"File already exists, skipping download: {dest_path}")
            return
        logging.info(f"Downloading file from {url} to {dest_path}")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logging.info("Download complete.")
        except Exception as e:
            logging.error(f"Error downloading file: {e}")
            sys.exit(1)

    def create_yaml_file(self):
        """Create a YAML file for dataset configuration.

        Returns:
            str: Path to the created YAML file.
        """
        dict_file = {
            "train": os.path.join(self.CONFIG.CUSTOM_DATASET_DIR, "train"),
            "val": os.path.join(self.CONFIG.CUSTOM_DATASET_DIR, "valid"),
            "test": os.path.join(self.CONFIG.CUSTOM_DATASET_DIR, "test"),
            "nc": self.CONFIG.NUM_CLASSES_TO_TRAIN,
            "names": self.CONFIG.CLASSES,
        }
        yaml_path = os.path.join(self.CONFIG.OUTPUT_DIR, "data.yaml")
        try:
            with open(yaml_path, "w") as file:
                yaml.dump(dict_file, file)
            logging.info(f"Created YAML file at {yaml_path}")
        except Exception as e:
            logging.error(f"Error creating YAML file: {e}")
            sys.exit(1)
        return yaml_path

    def read_yaml_file(self, file_path):
        """Read a YAML file and return its contents.

        Args:
            file_path (str): Path to the YAML file.

        Returns:
            dict: Contents of the YAML file or None if error occurs.
        """
        try:
            with open(file_path, "r") as file:
                data = yaml.safe_load(file)
            logging.info(f"Read YAML file: {file_path}")
            return data
        except Exception as e:
            logging.error(f"Error reading YAML file {file_path}: {e}")
            return None

    def print_yaml_data(self, data):
        """Print YAML data in a formatted way.

        Args:
            data (dict): YAML data to print.
        """
        formatted_yaml = yaml.dump(data, default_flow_style=False)
        logging.info("YAML data:\n" + formatted_yaml)

    def display_image(self, image_path, print_info=True, hide_axis=False):
        """Display an image using matplotlib.

        Args:
            image_path (str): Path to the image file.
            print_info (bool): Whether to print image info. Default is True.
            hide_axis (bool): Whether to hide plot axes. Default is False.
        """
        if not self.CONFIG.DISPLAY_IMAGES:
            return
        try:
            img = Image.open(image_path)
            if print_info:
                logging.info(f"Displaying image: {image_path} with size {img.size} and type {type(img)}")
            plt.imshow(img)
            if hide_axis:
                plt.axis("off")
            plt.show()
        except Exception as e:
            logging.error(f"Error displaying image {image_path}: {e}")

    def plot_random_images_from_folder(self, folder_path, num_images=20, seed=CONFIG.SEED):
        """Plot random images from a folder.

        Args:
            folder_path (str): Path to the folder containing images.
            num_images (int): Number of images to plot. Default is 20.
            seed (int): Random seed for reproducibility. Default is CONFIG.SEED.
        """
        if not self.CONFIG.DISPLAY_IMAGES:
            return
        try:
            random.seed(seed)
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".png", ".jpeg", ".gif"))]
            if len(image_files) < num_images:
                raise ValueError("Not enough images in the folder")
            selected_files = random.sample(image_files, num_images)
            num_cols = 5
            num_rows = (num_images + num_cols - 1) // num_cols
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
            axes = axes.flatten()
            for i, file_name in enumerate(selected_files):
                img_path = os.path.join(folder_path, file_name)
                img = Image.open(img_path)
                axes[i].imshow(img)
                axes[i].axis("off")
            for i in range(num_images, len(axes)):
                fig.delaxes(axes[i])
            plt.tight_layout()
            plt.show()
            logging.info("Plotted random images from folder.")
        except Exception as e:
            logging.error(f"Error plotting random images: {e}")

    def get_image_properties(self, image_path):
        """Get properties of an image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            dict: Dictionary containing image properties (width, height, channels, dtype).
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not read image file")
            properties = {
                "width": img.shape[1],
                "height": img.shape[0],
                "channels": img.shape[2] if len(img.shape) == 3 else 1,
                "dtype": img.dtype,
            }
            logging.info(f"Image properties for {image_path}: {properties}")
            return properties
        except Exception as e:
            logging.error(f"Error getting image properties for {image_path}: {e}")
            sys.exit(1)

    def dataset_statistics(self):
        """Compute and display dataset statistics."""
        try:
            class_idx = {str(i): self.CONFIG.CLASSES[i] for i in range(self.CONFIG.NUM_CLASSES_TO_TRAIN)}
            class_stat = {}
            data_len = {}
            class_info = []
            for mode in ["train", "valid", "test"]:
                class_count = {self.CONFIG.CLASSES[i]: 0 for i in range(self.CONFIG.NUM_CLASSES_TO_TRAIN)}
                labels_path = os.path.join(self.CONFIG.CUSTOM_DATASET_DIR, mode, "labels")
                for file in os.listdir(labels_path):
                    file_path = os.path.join(labels_path, file)
                    with open(file_path, "r") as f:
                        lines = f.readlines()
                        for cls in set([line.split()[0] for line in lines if line.strip()]):
                            class_count[class_idx[cls]] += 1
                data_len[mode] = len(os.listdir(labels_path))
                class_stat[mode] = class_count
                info = {"Mode": mode, **class_count, "Data_Volume": data_len[mode]}
                class_info.append(info)
            dataset_stats_df = pd.DataFrame(class_info)
            logging.info("Dataset statistics:\n" + dataset_stats_df.to_string())
            if self.CONFIG.DISPLAY_IMAGES:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                for i, mode in enumerate(["train", "valid", "test"]):
                    sns.barplot(
                        data=dataset_stats_df[dataset_stats_df["Mode"] == mode].drop(columns="Mode"),
                        ax=axes[i],
                        palette="Set2",
                    )
                    axes[i].set_title(f"{mode.capitalize()} Class Statistics")
                    axes[i].set_xlabel("Classes")
                    axes[i].set_ylabel("Count")
                    axes[i].tick_params(axis="x", rotation=90)
                    for p in axes[i].patches:
                        axes[i].annotate(
                            f"{int(p.get_height())}",
                            (p.get_x() + p.get_width() / 2.0, p.get_height()),
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="black",
                            xytext=(0, 5),
                            textcoords="offset points",
                        )
                plt.tight_layout()
                plt.show()
        except Exception as e:
            logging.error(f"Error computing dataset statistics: {e}")

    def base_model_inference(self, image_path):
        """Perform inference using the base model.

        Args:
            image_path (str): Path to the image for inference.
        """
        try:
            results = self.model.predict(
                source=image_path,
                classes=[0],
                conf=0.30,
                device=[0, 1],
                imgsz=(self.img_properties["height"], self.img_properties["width"]),
                save=True,
                save_txt=True,
                save_conf=True,
                exist_ok=True,
            )
            logging.info("Base model inference completed.")
            if self.CONFIG.DISPLAY_IMAGES:
                output_image = os.path.join(
                    self.CONFIG.OUTPUT_DIR, "runs", "detect", "predict", os.path.basename(image_path)
                )
                if os.path.exists(output_image):
                    self.display_image(output_image)
                else:
                    logging.warning(f"Inference output image not found at {output_image}")
        except Exception as e:
            logging.error(f"Error during base model inference: {e}")

    def train_model(self):
        """Train the YOLO model."""
        try:
            logging.info(f"Starting training with model weights: {self.CONFIG.BASE_MODEL_WEIGHTS}")
            self.model.train(
                data=self.yaml_path,
                task="detect",
                imgsz=(self.img_properties["height"], self.img_properties["width"]),
                epochs=self.CONFIG.EPOCHS,
                batch=self.CONFIG.BATCH_SIZE,
                optimizer=self.CONFIG.OPTIMIZER,
                lr0=self.CONFIG.LR,
                lrf=self.CONFIG.LR_FACTOR,
                weight_decay=self.CONFIG.WEIGHT_DECAY,
                dropout=self.CONFIG.DROPOUT,
                fraction=self.CONFIG.FRACTION,
                patience=self.CONFIG.PATIENCE,
                profile=self.CONFIG.PROFILE,
                label_smoothing=self.CONFIG.LABEL_SMOOTHING,
                name=f"{self.CONFIG.BASE_MODEL}_{self.CONFIG.EXP_NAME}",
                seed=self.CONFIG.SEED,
                val=True,
                amp=True,
                exist_ok=True,
                resume=False,
                device=[],
                verbose=False,
            )
            logging.info("Training completed.")
        except Exception as e:
            logging.error(f"Error during training: {e}")
            sys.exit(1)

    def export_model(self):
        """Export the trained model to ONNX format.

        Returns:
            str: Path to the exported model file.
        """
        try:
            export_path = self.model.export(
                format="onnx",
                imgsz=(self.img_properties["height"], self.img_properties["width"]),
                half=False,
                int8=False,
                simplify=False,
                nms=False,
            )
            logging.info(f"Model exported to {export_path}")
            return export_path
        except Exception as e:
            logging.error(f"Error during model export: {e}")
            return None

    def run(self):
        """Run the full finetuning pipeline."""
        logging.info("Script started.")

        # Setup directories
        self.create_directory(self.CONFIG.OUTPUT_DIR)
        self.create_directory(self.CONFIG.MODELS_DIR)
        self.create_directory(os.path.join(self.CONFIG.OUTPUT_DIR, "runs"))
        self.create_directory(os.path.join(self.CONFIG.OUTPUT_DIR, "runs", "detect", "predict"))

        # Download weights
        weights_url = f"https://github.com/ultralytics/assets/releases/download/v8.1.0/{self.CONFIG.BASE_MODEL}.pt"
        self.download_file(weights_url, self.CONFIG.BASE_MODEL_WEIGHTS)

        # YAML setup
        self.yaml_path = self.create_yaml_file()
        yaml_data = self.read_yaml_file(self.yaml_path)
        if yaml_data:
            self.print_yaml_data(yaml_data)

        # Image handling
        train_images_folder = os.path.join(self.CONFIG.CUSTOM_DATASET_DIR, "train", "images")
        if os.path.exists(train_images_folder):
            image_files = [f for f in os.listdir(train_images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if image_files:
                example_image_path = os.path.join(train_images_folder, random.choice(image_files))
                self.display_image(example_image_path, print_info=True, hide_axis=False)
            else:
                logging.warning("No image files found in training folder")
                example_image_path = None
        else:
            logging.warning(f"Training images folder not found: {train_images_folder}")
            example_image_path = None

        if os.path.exists(train_images_folder):
            self.plot_random_images_from_folder(train_images_folder)

        if os.path.exists(example_image_path):
            self.img_properties = self.get_image_properties(example_image_path)
        else:
            logging.error("Cannot get image properties; example image not found.")
            sys.exit(1)

        # Dataset stats and model loading
        self.dataset_statistics()
        logging.info(f"Loading YOLO model from {self.CONFIG.BASE_MODEL_WEIGHTS}")
        self.model = YOLO(self.CONFIG.BASE_MODEL_WEIGHTS)

        # Inference, training, and export
        self.base_model_inference(example_image_path)
        self.train_model()
        exported_model_path = self.export_model()

        logging.info("Script completed successfully.")
        return exported_model_path


if __name__ == "__main__":
    finetuner = Finetuning()
    exported_path = finetuner.run()
    if exported_path:
        logging.info(f"Model ready for reuse at: {exported_path}")