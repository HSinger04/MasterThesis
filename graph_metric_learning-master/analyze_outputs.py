import json
from argparse import ArgumentParser
import os
import sys
from collections import OrderedDict


def main(outputs_dir):
    result_collector = OrderedDict()
    for dirs in os.listdir(outputs_dir):
        results_path = os.path.join(outputs_dir, dirs, "test_results.json")
        try:
            with open(results_path, "r") as f:
                result_collector[dirs] = json.load(f)
        except:
            print("Error @ " + dirs, file=sys.stderr)
    metrics_to_analyze = ["accuracy"]
    for metric in metrics_to_analyze:
        ordered_by_metric = sorted(result_collector, key=lambda x: result_collector[x]["test"][metric])
        ordered_by_metric = {k:v for k, v in zip(ordered_by_metric, [result_collector[cfg]["test"][metric] for cfg in ordered_by_metric])}
        print(metric + ":")
        print(ordered_by_metric)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--outputs_dir', type=str, help='Directory path where the outputs to analyze are stored')
    main(**vars(parser.parse_args()))