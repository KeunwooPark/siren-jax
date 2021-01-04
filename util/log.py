import pathlib
import pickle
import json
import os
from collections import defaultdict
from util.plot import plot_losses
from PIL import Image
import numpy as np


def get_root_path():
    return pathlib.Path(__file__).parents[1]


class Logger:
    def __init__(self, name, create_if_exists=True):
        self.name = name
        self.create_if_exists = create_if_exists
        self.create_paths(name)
        self.log_file = self.get_log_file()
        self.log_for_plot = defaultdict(list)

    def create_paths(self, name):
        this_file = pathlib.Path(os.path.abspath(__file__))
        proj_root = this_file.parents[1]
        results_root = proj_root.joinpath("results")
        results_root.mkdir(exist_ok=True)

        result_path = results_root.joinpath(name)

        dup_cnt = 0
        while result_path.exists() and self.create_if_exists:
            dup_cnt += 1
            new_name = "{}{}".format(name, dup_cnt)
            result_path = results_root.joinpath(new_name)

        result_path.mkdir(exist_ok=True)
        self.result_path = result_path

    def get_log_file(self):
        loss_path = self.result_path / "logges.txt"
        return open(str(loss_path), "w")

    def save_option(self, option):
        setting_path = self.result_path / "option.txt"
        with open(str(setting_path), "w") as f:
            json_obj = json.dumps(option)
            f.write(json_obj)

    def save_net_params(self, params):
        model_path = self.result_path / "params.pkl"
        with open(str(model_path), "wb") as f:
            pickle.dump(params, f)

    def save_log(self, log):
        json_obj = json.dumps(log)
        self.log_file.write(json_obj + "\n")

        for k, v in log.items():
            self.log_for_plot[k].append(v)

    def save_image(self, name, img):
        if not isinstance(img, Image.Image):
            img = img.squeeze()
            img = Image.fromarray(np.uint8(img))
        img_path = self.result_path / (name + ".png")
        img.save(str(img_path))

    def get_plot_file_name(self):
        return str(self.result_path / "loss.png")

    def save_losses_plot(self):
        plot_losses(self.log_for_plot, self.get_plot_file_name())


class Loader:
    def __init__(self, name):
        self.create_path(name)

    def create_path(self, name):
        this_file = pathlib.Path(os.path.abspath(__file__))
        proj_root = this_file.parents[1]
        results_root = proj_root.joinpath("results")

        self.results_path = results_root.joinpath(name)

        if not self.results_path.exists():
            raise ValueError("No data with name {}".format(name))

    def load_option(self):
        option_path = self.results_path / "option.txt"
        option = None
        with open(str(option_path)) as f:
            lines = f.readlines()
            option = json.loads(lines[0])  # there should be only one line

        return option

    def load_params(self):
        model_path = self.results_path.joinpath("params.pkl")
        model = None
        with open(str(model_path), "rb") as f:
            model = pickle.load(f)

        return model

    def get_image_filename(self, name):
        image_path = self.results_path / (name + ".png")
        return str(image_path)

    def load_pil_image(self, name):
        fn = self.get_image_filename(name)
        return Image.open(fn)

    def load_log(self):
        loss_path = self.results_path.joinpath("logges.txt")
        losses = None
        with open(str(loss_path), "r") as f:
            lines = f.readlines()
            losses = self._parse_logges(lines)

        return losses

    def _parse_logges(self, lines):
        logges = defaultdict(list)
        for line in lines:
            json_obj = json.loads(line)
            for k, v in json_obj.items():
                logges[k].append(v)

        return logges
