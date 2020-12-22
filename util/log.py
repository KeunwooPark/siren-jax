import pathlib
import pickle
import json
import os
from collections import defaultdict

def get_root_path():
    return pathlib.Path(__file__).parents[1]

class Logger:
    def __init__(self, name):
        self.name = name
        self.create_paths(name)
        self.log_file = self.get_log_file()

    def create_paths(self, name):
        this_file = pathlib.Path(os.path.abspath(__file__))
        proj_root = this_file.parents[1]
        results_root = proj_root.joinpath("results")
        results_root.mkdir(exist_ok=True)

        result_path = results_root.joinpath(name)

        dup_cnt = 0
        while result_path.exists():
            dup_cnt += 1
            new_name = "{}{}".format(name, dup_cnt)
            result_path = results_root.joinpath(new_name)

        result_path.mkdir(exist_ok=True)
        self.result_path = result_path

    def get_log_file(self):
        loss_path = self.result_path.joinpath("logges.txt")
        return open(str(loss_path), "w")

    def save_option(self, option):
        setting_path = self.result_path.joinpath("option.txt")
        with open(str(setting_path), "w") as f:
            yaml_obj = json.dumps(option)
            f.write(yaml_obj)

    def save_model(self, model):
        model_path = self.result_path.joinpath("model.pkl")
        with open(str(model_path), "wb") as f:
            pickle.dump(model, f)

    def save_log(self, log):
        json_obj = json.dumps(log)
        self.log_file.write(json_obj + "\n")


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

        self._create_snapshot_path()

    def _create_snapshot_path(self):
        self.snapshot_path = self.results_path / "snapshots"

    def load_model(self):
        model_path = self.results_path.joinpath("model.pkl")
        model = None
        with open(str(model_path), "rb") as f:
            model = pickle.load(f)

        return model

    def load_option(self):
        option_path = self.results_path.joinpath("option.yaml")
        return load_yaml_option(option_path)

    def load_snapshot(self, step):
        ss_path = self.snapshot_path / ("snapshot_{}.pkl".format(step))
        snapshot = None
        with open(str(ss_path), "rb") as f:
            snapshot = pickle.load(f)

        return snapshot

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


def load_yaml_option(file_name):
    option = None
    with open(file_name, "r") as f:
        option = yaml.load(f, Loader=yaml.FullLoader)

    return option


def load_base_option():
    this_file = pathlib.Path(os.path.abspath(__file__))
    proj_root = this_file.parents[1]
    option_path = proj_root / "base_option.yaml"
    return load_yaml_option(str(option_path))
