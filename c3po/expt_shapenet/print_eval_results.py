import argparse
import sys
import torch
import yaml
import pickle
import csv

sys.path.append('../..')


def write_to_csv(dict_data, csv_file, csv_columns):
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)
    except IOError:
        print("I/O error")


if __name__ == "__main__":
    """
    python print_eval_results.py --folder "./temp" --objects $OBJECT_LIST --filename "results"
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder",
                        type=str,
                        default='./temp')
    parser.add_argument("--objects",
                        type=str,
                        default=' ')
    parser.add_argument("--filename",
                        type=str,
                        default='results')

    args = parser.parse_args()

    folder_ = args.folder
    objects = args.objects.split(' ')
    dict_data = []
    for i, object in enumerate(objects):
        save_file = open(args.folder + "/" + object + ".pkl", 'rb')
        data = pickle.load(save_file)

        if i == 0:
            csv_columns = data.keys()

        dict_data.append(data)

    write_to_csv(dict_data=dict_data, csv_file=f'{str(args.filename)}.csv', csv_columns=csv_columns)

