import logging

import torch
from torch.utils.data import DataLoader
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from iiot_fl.config import build_model_config, build_train_config
from iiot_fl.dataset import INPUT_DIM, load_partition
from iiot_fl.model import IIoTFLNet, DualTaskLoss
from iiot_fl.task import get_parameters, set_parameters, train, evaluate

MACHINE_TYPES = [
    "Furnace",
    "Compressor",
    "Labeler",
    "Boiler",
    "Laser_Cutter",
    "Robot_Arm",
    "Crane",
    "Forklift_Electric",
    "3D_Printer",
    "Vacuum_Packer",
    "CMM",
    "Injection_Molder",
    "Grinder",
    "Pick_and_Place",
    "Dryer",
    "Conveyor_Belt",
    "Shrink_Wrapper",
    "Valve_Controller",
    "Palletizer",
    "Automated_Screwdriver",
    "Press_Brake",
    "Carton_Former",
    "XRay_Inspector",
    "Vision_System",
    "CNC_Mill",
    "CNC_Lathe",
    "Heat_Exchanger",
    "Hydraulic_Press",
    "Industrial_Chiller",
    "Shuttle_System",
    "AGV",
    "Pump",
    "Mixer",
]

logger = logging.getLogger(__name__)


class Client(NumPyClient):
    def __init__(
        self,
        model: IIoTFLNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: DualTaskLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        local_epochs: int,
        machine_type: str,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.local_epochs = local_epochs
        self.machine_type = machine_type

    def get_parameters(self, config: dict) -> list:
        return get_parameters(self.model)

    def set_parameters(self, parameters: list):
        set_parameters(self.model, parameters)

    def fit(self, parameters: list, config: dict) -> tuple:
        self.set_parameters(parameters)

        epochs = int(config.get("local_epochs", self.local_epochs))
        logger.info(
            "[%s] Starting local trainging (%d epochs).", self.machine_type, epochs
        )

        metrics = train(
            self.model,
            self.train_loader,
            self.criterion,
            self.optimizer,
            self.scheduler,
            self.device,
            epochs,
        )

        num_samples = len(self.train_loader.dataset)
        logger.info(
            "[%s] fit complete | samples=%d | loss=%.4f",
            self.machine_type,
            num_samples,
            metrics["avg_loss"],
        )

        return self.get_parameters({}), num_samples, metrics

    def evaluate(self, parameters: list, config: dict) -> tuple:
        self.set_parameters(parameters)

        loss, num_samples, metrics = evaluate(
            self.model, self.val_loader, self.criterion, self.device
        )

        logger.info(
            "[%s] evaluate complete | samples=%d | loss=%.4f | f1=%.4f",
            self.machine_type,
            num_samples,
            loss,
            metrics["fail_f1"],
        )

        return loss, num_samples, metrics


def client_fn(context: Context) -> Client:
    partition_id = context.node_config["partition-id"]
    machine_type = MACHINE_TYPES[partition_id % len(MACHINE_TYPES)]
    data_dir = context.node_config.get(
        "data-dir",
        context.run_config.get("data.data-dir", "/data"),
    )

    logger.info("Initialising client for machine type: %s", machine_type)

    model_config = build_model_config(context.run_config)
    train_config = build_train_config(context.run_config)

    model_config["input_dim"] = INPUT_DIM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    train_loader, val_loader, pos_weight = load_partition(
        data_dir,
        machine_type,
        train_config["batch_size"],
    )

    model = IIoTFLNet(model_config).to(device)
    criterion = DualTaskLoss(
        rul_weight=train_config["rul_loss_weight"],
        fail_weight=train_config["failure_loss_weight"],
    )
    criterion.set_pos_weight(pos_weight.to(device))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["lr"],
        weight_decay=train_config["weight_decay"],
    )

    local_epochs = train_config["local_epochs"]
    steps_per_round = len(train_loader) * local_epochs
    if train_config["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=steps_per_round,
            eta_min=train_config["lr"] * 0.01,
        )
    else:
        scheduler = None

    return Client(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        local_epochs,
        machine_type,
    ).to_client()


app = ClientApp(client_fn=client_fn)
