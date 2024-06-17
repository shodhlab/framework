from model import Transformer
from config_reader import Config
from custom.inference import Inference


config = Config()
inference_engine = Inference(Transformer, config)
inference_engine.infer("first citizen :", 100)
