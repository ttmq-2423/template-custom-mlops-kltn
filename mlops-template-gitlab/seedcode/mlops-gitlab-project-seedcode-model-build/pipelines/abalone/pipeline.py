"""Example workflow pipeline script for medical_mae pipeline.

Implements a get_pipeline(**kwargs) method.
"""
import os

import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.pytorch import PyTorch

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
    """Gets the sagemaker client.

    Args:
        region: the aws region to start the session

    Returns:
        `sagemaker.session.Session instance
    """
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )


def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    model_package_group_name="MedicalMAEPackageGroup",
    pipeline_name="MedicalMAE_Pipeline",
    base_job_prefix="MedicalMAE",
):
    """Gets a SageMaker ML Pipeline instance working with on medical_mae data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    # parameters for pipeline execution
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value="ml.c5.xlarge"
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType", default_value="ml.c5.xlarge"
    )
    input_data_url = ParameterString(
        name="InputDataUrl",
        default_value="s3://mqht/medical_mae_mixi/"
    )
    train_image_uri = ParameterString(
        name="TrainImageUri",
        default_value="600627364468.dkr.ecr.us-east-1.amazonaws.com/mq/train_image:latest"
    )
    processing_image_uri = ParameterString(
        name="ProcessingImageUri",
        default_value="600627364468.dkr.ecr.us-east-1.amazonaws.com/mq/processing:latest"
    )
    evaluate_image_uri = ParameterString(
        name="EvaluateImageUri",
        default_value="600627364468.dkr.ecr.us-east-1.amazonaws.com/evaluate:latest"
    )
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    mse_threshold = ParameterInteger(name='MseThreshold', default_value=50)


    # Processing step for data preprocessing
    script_processor = ScriptProcessor(
        role=role,
        image_uri=processing_image_uri,
        command=["python3"],
        instance_count=1,
        instance_type=processing_instance_type,
        sagemaker_session=sagemaker_session,
        base_job_name=f"{base_job_prefix}/preprocess-medical-mae",
    )
    input_data = ProcessingInput(
        source=input_data_url,
        destination="/opt/ml/processing/input",
        input_name="input-data",
    )
    output_data = ProcessingOutput(
        source="/opt/ml/processing/output",
        destination=f"s3://{default_bucket}/data_train",
        output_name="output-data",
    )
    step_process = ProcessingStep(
        name="PreprocessStep",
        processor=script_processor,
        inputs=[input_data],
        outputs=[output_data],
        code="pipelines/abalone/preprocess_script.py",
    )

    # Training step for model training
    estimator = PyTorch(
        entry_point="pipelines/abalone/train_script.py",
        source_dir="./",
        role=role,
        instance_count=1,
        instance_type=training_instance_type,
        image_uri=train_image_uri,
        script_mode=True,
        region=region,
        output_path=f"s3://{default_bucket}/output/train",
        model_output_path=f"s3://{default_bucket}/model",
        sagemaker_session=sagemaker_session,
        base_job_name=f"{base_job_prefix}/train-medical-mae",
    )
    train_input_data = TrainingInput(s3_data=f"s3://{default_bucket}/data_train")
    source_data = TrainingInput(s3_data=input_data_url)
    step_train = TrainingStep(
        name="TrainModelStep",
        estimator=estimator,
        inputs={"training": train_input_data, "code": source_data},
    )


# Evaluation step for model evaluation
    evaluator = PyTorch(
        entry_point="pipelines/abalone/evaluate_script.py",
        source_dir="./",
        role=role,
        instance_count=1,
        instance_type=training_instance_type,
        image_uri=evaluate_image_uri,
        script_mode=True,
        region=region,
        output_path=f"s3://{default_bucket}/output/evaluate",
        model_output_path=f"s3://{default_bucket}/model",
        sagemaker_session=sagemaker_session,
        base_job_name=f"{base_job_prefix}/evaluate-medical-mae",
    )
    eval_input_data = TrainingInput(s3_data=f"s3://{default_bucket}/data_train")
    eval_source_data = TrainingInput(s3_data=input_data_url)
    step_eval = TrainingStep(
       name="EvaluateModelStep",
        estimator=evaluator,
        inputs={"training": eval_input_data, "code": eval_source_data},
    )

    # Get evaluation report from Evaluate Step
    evaluation_report = PropertyFile(
        name="MedicalMAEEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    # register model step that will be conditionally executed
  

    model_metrics = ModelMetrics(
     model_statistics=MetricsSource(
         s3_uri="s3://evaluate-output/output/evaluation.json",
         content_type="application/json"
        )
    )

    
    

    step_register = RegisterModel(
        name="RegisterMedicalMAEModel",
        estimator=estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics, # Pass the model metrics object
    )
      # condition step for evaluating model quality and branching execution
   


    # Define dependency
    step_train.add_depends_on([step_process])
    step_eval.add_depends_on([step_train])
    #step_register.add_depends_on([step_eval])

    # Pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
           processing_instance_type,
           training_instance_type,
           input_data_url,
           train_image_uri,
           processing_image_uri,
           evaluate_image_uri,
           model_approval_status,
            mse_threshold,
        ],
        steps=[step_process, step_train, step_eval,step_register],
        sagemaker_session=sagemaker_session,
    )
    return pipeline
