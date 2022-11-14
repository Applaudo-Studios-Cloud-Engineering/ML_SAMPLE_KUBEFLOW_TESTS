import kfp
import kfp.dsl as dsl
from kfp import compiler
from nodes import create_dataset, create_feature_engineering_pipeline, create_ml_pipeline_classifier, \
    create_ml_pipeline_regressor, create_ml_pipeline_extra_classifier, create_ml_pipeline_extra_regressor, \
    evaluate_accuracy
from typing import List


@dsl.pipeline(
    name='preprocessing-pipeline',
    description='An example pipeline that creates the dataset.',
    pipeline_root='gs://data-bucket-6929d24320ef4e55/dataTrain/build'
)
def preprocessing_pipeline(path: str = 'gs://data-bucket-6929d24320ef4e55/dataTrain/train.csv'):
    dataset_task = create_dataset(path)
    # print(dataset_task.outputs["test"])
    task = create_feature_engineering_pipeline(dataset_task.outputs["dataset"])
    first_acc = create_ml_pipeline_classifier(path=task.outputs["dataset_feature_engineering"])
    second_acc = create_ml_pipeline_regressor(path=task.outputs["dataset_feature_engineering"])
    third_acc = create_ml_pipeline_extra_classifier(
        path=task.outputs["dataset_feature_engineering"])
    fourth_acc = create_ml_pipeline_extra_regressor(
        path=task.outputs["dataset_feature_engineering"])

    winner = evaluate_accuracy(first_acc.output, second_acc.output, third_acc.output, fourth_acc.output)


compiler.Compiler(mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE).compile(pipeline_func=preprocessing_pipeline,
                                                                            package_path='pipeline.yaml')
