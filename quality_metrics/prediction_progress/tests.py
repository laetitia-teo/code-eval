import json
from quality_metrics.common import dataset_from_p3, Problem
from quality_metrics.prediction_progress.pp import PredictionProgressCE


def test_init(model_path, dataset, reference_problem):
    return PredictionProgressCE(
        model_id_or_path=model_path,
        archive_path_or_list=dataset,
        reference_problem=reference_problem,
    )


def test_init_masked_sol(model_path, dataset, reference_problem):
    return PredictionProgressCE(
        model_id_or_path=model_path,
        archive_path_or_list=dataset,
        reference_problem=reference_problem,
        solution_mask=True,
    )


def test_losses(pp_metric):
    print(pp_metric.original_losses)


def test_differences(pp_metric, problem):
    print(pp_metric.differences(problem))


def test_solution_masking():
    pass


if __name__ == "__main__":
    # load simplified archive
    with open('data/dataset.json', 'r') as f:
        dataset = json.load(f)
    dataset = dataset[:20]
    problem_dataset = dataset_from_p3(dataset)

    # reference puzzle
    REF_PUZZLE_NODOC = '''def sat(s: List[str]):
    return len(set(s)) == 1000 and all((x.count("a") > x.count("b")) and ('b' in x) for x in s)'''

    REF_SOL = '''def sol():
        return ["a" * (i + 2) + "b" for i in range(1000)]'''

    ref_problem = Problem(idx='ref_problem', instruction=REF_PUZZLE_NODOC, completion=REF_SOL)

    # tests 
    pp_metric = test_init(
        model_path='deepseek-ai/deepseek-coder-1.3b-instruct',
        dataset=problem_dataset,
        reference_problem=ref_problem,
    )
    test_losses(pp_metric)
    test_differences(pp_metric, problem=problem_dataset[0])

    # solution masking
    pp_metric = test_init_masked_sol(
        model_path='deepseek-ai/deepseek-coder-1.3b-instruct',
        dataset=problem_dataset,
        reference_problem=ref_problem,
    )
    test_losses(pp_metric)
    test_differences(pp_metric, problem=problem_dataset[0])
    print(pp_metric._get_losses(problem_dataset[0]))
