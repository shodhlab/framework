import torch
from model import Transformer
from config_read import Config
from preprocess import DataModule
from utils.misc import measure_time

# import logging

from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    config = Config()
    start_time = measure_time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = TensorBoardLogger("logs/", name="transformer")
    version = logger.version
    profiler_log_dir = f"logs/profiler/version_{version}"
    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_log_dir),
        trace_memory=True,
        export_to_chrome=True,
    )
    strategy = DeepSpeedStrategy(config=config.deepspeed)
    dataModule = DataModule(config.train, config.preprocess)

    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=config.train["max_epochs"],
        min_epochs=config.train["min_epochs"],
        precision="bf16-true",
        log_every_n_steps=config.train["log_steps"],
        strategy=strategy,
        logger=logger,
        profiler=profiler,
        callbacks=[lr_monitor],
    )

    print(f"[{measure_time(start_time)}]Loading data on {trainer.global_rank}...")
    dataModule.setup()
    print(f"[{measure_time(start_time)}]Data loaded on {trainer.global_rank}.")

    print(f"[{measure_time(start_time)}]Initializing model on {trainer.global_rank}...")
    model = Transformer(config.train, dataModule.vocab_size).to(device)
    model = model.double()
    print(f"[{measure_time(start_time)}]Model initialized on {trainer.global_rank}.")

    print(f"[{measure_time(start_time)}]Starting training on {trainer.global_rank}...")
    trainer.fit(model, dataModule)
    print(f"[{measure_time(start_time)}]Training complete on {trainer.global_rank}.")

    print(f"[{measure_time(start_time)}]Starting testing on {trainer.global_rank}...")
    trainer.test(model, dataModule)
    print(f"[{measure_time(start_time)}]Testing complete on {trainer.global_rank}.")
