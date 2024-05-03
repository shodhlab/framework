import torch
from model import Transformer
import pytorch_lightning as pl
from preprocess import DataModule
from utils.misc import measure_time, load_configs

# from pytorch_lightning.strategies import DeepSpeedStrategy
# from pytorch_lightning.callbacks import LearningRateMonitor
# from pytorch_lightning.profilers import PyTorchProfiler


torch.set_float32_matmul_precision("high")
# lr_monitor = LearningRateMonitor(logging_interval="step")
preprocess_config, train_config, deepspeed_config = load_configs()
start_time = measure_time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = pl.loggers.TensorBoardLogger("logs/", name="transformer")
# profiler = PyTorchProfiler(
#     on_trace_ready=torch.profiler.tensorboard_trace_handler("logs/profiler"),
#     trace_memory=True,
#     schedule=torch.profiler.schedule(wait=1, warmup=1, active=20),
# )
# strategy = DeepSpeedStrategy(accelerator="auto", zero_optimization=True, stage=3)

dataModule = DataModule(train_config, preprocess_config)
trainer = pl.Trainer(
    accelerator="auto",
    devices="auto",
    max_epochs=train_config["max_epochs"],
    min_epochs=train_config["min_epochs"],
    precision="bf16-true",
    log_every_n_steps=train_config["log_steps"],
    # strategy=strategy,
    logger=logger,
    # profiler=profiler,
    # callbacks=[lr_monitor],
)

print(f"[{measure_time(start_time)}]Loading data on {trainer.global_rank}...")
dataModule.setup()
print(f"[{measure_time(start_time)}]Data loaded on {trainer.global_rank}.")

print(f"[{measure_time(start_time)}]Initializing model on {trainer.global_rank}...")
model = Transformer(train_config, dataModule.vocab_size).to(device)
print(f"[{measure_time(start_time)}]Model initialized on {trainer.global_rank}.")

print(f"[{measure_time(start_time)}]Starting training on {trainer.global_rank}...")
trainer.fit(model, dataModule)
print(f"[{measure_time(start_time)}]Training complete on {trainer.global_rank}.")
