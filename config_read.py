import os
import json


class Config:
    def __init__(self):
        self.config_files = self.get_files()
        self.preprocess, self.train, self.deepspeed = self.loads()
        # Process these to actual python objects

    def get_files(self):
        config_folder = "./config"
        config_files = []
        for file in os.listdir(config_folder):
            if file.endswith(".json"):
                config_files.append(os.path.join(config_folder, file))
        return config_files

    def loads(self):
        preprocess = {}
        train = {}
        deepspeed = {}
        for file in self.config_files:
            with open(file, "r") as f:
                config = json.load(f)
                if "preprocess" in file:
                    preprocess = config
                elif "train" in file:
                    train = config
                elif "deepspeed" in file:
                    deepspeed = config
        return preprocess, train, deepspeed
