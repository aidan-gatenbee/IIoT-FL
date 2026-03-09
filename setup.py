from generate_compose import generate_compose

import kagglehub
import pandas as pd
import os
import sys
import subprocess
import argparse


def setup_and_split():
    # Load the dataset the first time

    path = kagglehub.dataset_download("canozensoy/industrial-iot-dataset-synthetic")
    data = path + "/factory_sensor_simulator_2040.csv"

    df = pd.read_csv(data)

    # Split the dataframe into 33 partitions based on the "Machine_Type" column
    partitions = {}
    for machine_type in df["Machine_Type"].unique():
        partitions[machine_type] = df[df["Machine_Type"] == machine_type]

    # Save each partition into a separate directory in ./data/{machine_type}
    for machine_type, partition_df in partitions.items():
        os.makedirs(f"./data/{machine_type}", exist_ok=True)
        partition_df.to_csv(f"./data/{machine_type}/data.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build images before starting services.",
    )
    args = parser.parse_args()

    build_selected = args.build

    if "--build" not in sys.argv:
        build_prompt = (
            input("Build Docker images before startup? [y/N]: ").strip().lower()
        )
        build_selected = build_prompt in {"y", "yes"}

    if build_selected:
        build_confirm = (
            input(
                "Warning: Building Docker images can take around 2 hours. Are you sure you want to proceed? [y/N]: "
            )
            .strip()
            .lower()
        )
        if build_confirm not in {"y", "yes"}:
            print("Aborting setup.")
            sys.exit(0)

    # Only run the setup if the data directory doesn't exist
    if not os.path.exists("./data"):
        setup_and_split()

    generate_compose("./data", "docker-compose.generated.yml")
    command = ["docker", "compose", "-f", "docker-compose.generated.yml", "up"]
    if build_selected:
        command.append("--build")

    subprocess.run(command)
