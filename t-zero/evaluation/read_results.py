import argparse
import json
import os

import numpy as np

from run_eval import get_task_name
from template_list import template_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default='./debug')
    parser.add_argument("--model_name_or_path", default='t5-base')
    args = parser.parse_args()

    accs = []
    for dataset, config in template_list:
        tmp_accs = []
        for template in template_list[(dataset, config)]:
            task_name = get_task_name(dataset, config, template)
            output_path = os.path.join(args.output_dir, task_name + ".json")
            if not os.path.exists(output_path):
                print(output_path, "does not exist!")
                tmp_accs.append(0.0)
                continue
            with open(output_path, "r", encoding="utf-8") as fin:
                results = json.load(fin)
                acc = float(results['evaluation']['accuracy'])
            tmp_accs.append(acc)
        tmp_acc = np.mean(tmp_accs)
        accs.append(tmp_acc)
        print(dataset, config, tmp_acc)

    print('overall Acc:', np.mean(accs))


if __name__ == '__main__':
    main()
