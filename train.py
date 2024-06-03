import torch
from model import Transformer
import pytorch_lightning as pl
from preprocess import DataModule
from config_read import Config
from utils.misc import measure_time
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.profilers import PyTorchProfiler

# from lightning.fabric.utilities.throughput import ThroughputMonitor, measure_flops
# from lightning.fabric.loggers import TensorBoardLogger
# from pytorch_lightning.callbacks import ModelCheckpoint


torch.set_float32_matmul_precision("high")
lr_monitor = LearningRateMonitor(logging_interval="step")
# config.preprocess, config.train, deepspeed_config = load_configs()
config = Config()
start_time = measure_time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = pl.loggers.TensorBoardLogger("logs/", name="transformer")
version = logger.version
profiler_log_dir = f"logs/profiler/version_{version}"

profiler = PyTorchProfiler(
    on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_log_dir),
    trace_memory=True,
    export_to_chrome=True,
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=20, repeat=1),
)

strategy = DeepSpeedStrategy(config=config.deepspeed)
dataModule = DataModule(config.train, config.preprocess)

# throughput = ThroughputMonitor()


# checkpoint_path = "logs/checkpoint/epoch=11-step=2160-v1.ckpt"
# checkpoint_callback = ModelCheckpoint(
#     dirpath="logs/checkpoint",
#     monitor="val_loss",
#     verbose=True,
#     save_top_k=1,
#     mode="min",
# )

trainer = pl.Trainer(
    accelerator="auto",
    devices="auto",
    max_epochs=config.train["max_epochs"],
    min_epochs=config.train["min_epochs"],
    precision="bf16-true",
    log_every_n_steps=config.train["log_steps"],
    strategy=strategy,
    logger=logger,
    profiler=profiler,
    # callbacks=[lr_monitor, checkpoint_callback]
    callbacks=[lr_monitor],
)

print(f"[{measure_time(start_time)}]Loading data on {trainer.global_rank}...")
dataModule.setup()
print(f"[{measure_time(start_time)}]Data loaded on {trainer.global_rank}.")

# with torch.device("meta"):
#     meta_model = Transformer(config.train, dataModule.vocab_size)
#     x = torch.randint(
#         0, 1, (config.train["batch_size"], config.train["sequence_length"])
#     )
#     model_fwd = lambda: meta_model(x)
#     model_loss = lambda y: torch.nn.CrossEntropyLoss()(y, x)
#     measured_flops = measure_flops(meta_model, model_fwd, model_loss)
#     print(f"Measured TFLOPs: {measured_flops:.2f}")
#     del meta_model, x

print(f"[{measure_time(start_time)}]Initializing model on {trainer.global_rank}...")
model = Transformer(config.train, dataModule.vocab_size).to(device)
# model = Transformer.load_from_checkpoint(
#     checkpoint_path, config=config.train, vocab_size=dataModule.vocab_size
# ).to(device)
print(f"[{measure_time(start_time)}]Model initialized on {trainer.global_rank}.")

print(f"[{measure_time(start_time)}]Starting training on {trainer.global_rank}...")
trainer.fit(model, dataModule)
print(f"[{measure_time(start_time)}]Training complete on {trainer.global_rank}.")

print(f"[{measure_time(start_time)}]Starting testing on {trainer.global_rank}.")
trainer.test(model, dataModule)
print(f"[{measure_time(start_time)}]Teasting complete on {trainer.global_rank}.")
