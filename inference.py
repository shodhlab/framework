from model import Transformer
from config_reader import Config
from custom.inference import Inference


config = Config()
inference_engine = Inference(Transformer, config)
print(inference_engine.infer("MENENIUS:\nEither you must", 200))
