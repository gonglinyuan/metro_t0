import argparse
import subprocess

from template_list import template_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='google/t5-xxl-lm-adapt')
    parser.add_argument("--task", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--out_dir")
    args = parser.parse_args()

    model_name_or_path = args.model

    base_command = [
        'python', 'run_eval.py',
        '--model_name_or_path', model_name_or_path,
        '--output_dir', args.out_dir,
        '--parallelize'
    ]

    # read template file 
    # --dataset_name super_glue --dataset_config_name rte --template_name "$i"
    for dataset, config in template_list:
        # break
        if args.task is not None and config != args.task:
            continue
        if args.dataset is not None and dataset != args.dataset:
            continue

        print(dataset, config)
        for template in template_list[(dataset, config)]:
            print('\t', template)
            if config is None:
                command = base_command + ['--dataset_name', dataset, '--template_name', template]
            else:
                command = base_command + ['--dataset_name', dataset,
                                          '--dataset_config_name', config,
                                          '--template_name', template]
            # print(command)

            subprocess.run(command)  # , stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            # exit()
        # exit()

    # print results 
    # python read_results.py --model_name_or_path google/t5-large-lm-adapt --output_dir ./debug
    eval_command = [
        'python', 'read_results.py',
        '--model_name_or_path', model_name_or_path,
        '--output_dir', args.out_dir
    ]
    subprocess.run(eval_command)
    # print(eval_command)
