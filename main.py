import os
import argparse
import scipy
import numpy as np
import pandas as pd
from model_zoo import *


def debug_model(model_cls=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--comment", default=None, type=str)
    if model_cls is None:
        parser.add_argument("--model", default="MNNMDA", choices=["MNNMDA", "LRLSHMDA", "NTSHMDA", "GATMDA", "KATZHMDA"])
    parser = Experiment.add_argparse_args(parser)
    options, other_args = parser.parse_known_args()
    model_cls = globals()[options.model]
    parser = model_cls.add_argparse_args(parser)
    config = parser.parse_args()
    experiment = Experiment(**vars(config))
    experiment.run(model_cls, comment=config.comment)


def search_params(model_cls=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--comment", default=None, type=str)
    if model_cls is None:
        parser.add_argument("--model", default="MNNMDA", choices=["MNNMDA", "LRLSHMDA", "NTSHMDA", "GATMDA", "KATZHMDA"])
    parser = Experiment.add_argparse_args(parser)
    options, other_args = parser.parse_known_args()
    model_cls = globals()[options.model]
    parser = model_cls.add_argparse_args(parser)
    config = parser.parse_args()
    experiment = Experiment(**vars(config))
    # experiment.run(model_cls, comment=config.comment, debug=True)

    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        for beta in [0.1, 1.0, 10.0, 100.0]:
            print(f"alpha:{alpha}, beta:{beta}")
            config.alpha = alpha
            config.beta = beta
            try:
                experiment = Experiment(**vars(config))
                experiment.run(model_cls, comment=f"search_params/{alpha}-{beta}", debug=True)
            except:
                pass
    experiment.collect_result(os.path.join(experiment.DEFAULT_DIR, "search_params"))


if __name__=="__main__":
    # search_params()
    debug_model()
    # save_dir = "/home/jm/PycharmProjects/yss/lcc/MNNMDA"
    # Experiment.collect_result(save_dir)
