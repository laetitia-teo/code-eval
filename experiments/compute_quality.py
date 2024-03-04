# temporary version
import json
import pathlib
import hydra
from tqdm import tqdm
from quality_metrics.common import dataset_from_p3, save_dataset
from quality_metrics.prediction_progress.pp import PredictionProgressCE


def compute_quality(metric):
    pass


@hydra.main(config_path='conf', config_name='quality_default')
def main(args):
    match args.metric.name:
        case 'pp':
            metric = PredictionProgressCE(
                **args.metric,
            )
        case _:
            raise NotImplementedError(f'{args.metric.name} metric not implemented')
    pass

    match args.dataset.name:
        case 'p3':
            json_dataset = json.load(open(args.dataset.path, 'r'))
            dataset = dataset_from_p3(json_dataset)
        case _:
            raise NotImplementedError(f'{args.dataset.name} dataset not implemented')

    path_name = pathlib.Path(args.metric.archive_path_or_list).stem
    save_path = args.dataset.path.replace(
        '.json', 
        f'_quality_{path_name}.json'
    )

    # TODO change this
    for i, p in enumerate(tqdm(dataset)):
        if len(p.instruction) + len(p.completion) < 1500:
            diff = metric.differences(p, return_list=True)
            quality = diff
            p.quality = quality
        if i + 1 % args.save_every:
            save_dataset(dataset, save_path)

    save_dataset(dataset, save_path)

if __name__ == "__main__":
    main()