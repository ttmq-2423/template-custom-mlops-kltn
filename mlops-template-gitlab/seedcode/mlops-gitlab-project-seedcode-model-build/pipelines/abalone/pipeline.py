"""Example workflow pipeline script for medical_mae pipeline.

Implements a get_pipeline(**kwargs) method.ddddddg
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
from sagemaker.workflow.parameters import ParameterFloat
from sagemaker.model import Model
from sagemaker.model_metrics import ModelMetrics
from sagemaker.inputs import TrainingInput

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker_client


def get_session(region, default_bucket):
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline(
    region,
    role=None,
    default_bucket=None,
    model_package_group_name="MedicalMAEPackageGroup",
    pipeline_name="MedicalMAE_Pipeline",
    base_job_prefix="MedicalMAE",
    **kwargs,  # Bỏ qua tham số không mong muốn
):
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.c5.xlarge")
    training_instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.c5.xlarge")
    input_data_url = ParameterString(name="InputDataUrl", default_value="s3://mqht/medical_mae_mixi/")
    train_image_uri = ParameterString(name="TrainImageUri", default_value="600627364468.dkr.ecr.us-east-1.amazonaws.com/mq/train_image:latest")
    processing_image_uri = ParameterString(name="ProcessingImageUri", default_value="600627364468.dkr.ecr.us-east-1.amazonaws.com/mq/processing:latest")
    evaluate_image_uri = ParameterString(name="EvaluateImageUri", default_value="600627364468.dkr.ecr.us-east-1.amazonaws.com/evaluate:latest")
    model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")
    auc_threshold = ParameterFloat(name='AucThreshold', default_value=0.5)

    script_processor = ScriptProcessor(
        role=role,
        image_uri=processing_image_uri,
        command=["python3"],
        instance_count=1,
        instance_type=processing_instance_type,
        sagemaker_session=sagemaker_session,
        base_job_name=f"{base_job_prefix}/preprocess-medical-mae",
    )
    input_data = ProcessingInput(source=input_data_url, destination="/opt/ml/processing/input", input_name="input-data")
    output_data = ProcessingOutput(source="/opt/ml/processing/output", destination=f"s3://{default_bucket}/data_train", output_name="output-data")
    step_process = ProcessingStep(name="PreprocessStep", processor=script_processor, inputs=[input_data], outputs=[output_data], code="pipelines/abalone/preprocess_script.py")

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
    step_train = TrainingStep(name="TrainModelStep", estimator=estimator, inputs={"training": train_input_data, "code": source_data})
    step_train.add_depends_on([step_process])

    evaluation_processor = ScriptProcessor(
        role=role,
        image_uri=evaluate_image_uri,
        command=["python3"],
        instance_count=1,
        instance_type=processing_instance_type,
        sagemaker_session=sagemaker_session,
        base_job_name=f"{base_job_prefix}/evaluate-medical-mae",
    )
    evaluation_report = PropertyFile(name="EvaluationReport", output_name="evaluation", path="evaluation.json")
    step_evaluate = ProcessingStep(
        name="EvaluateModelStep",
        processor=evaluation_processor,

        inputs=[
        ProcessingInput(  
            source=input_data_url,  
            destination="/opt/ml/processing/code"  
        ),
        ProcessingInput( 
            source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model"
        ),
    ],        
        outputs=[ProcessingOutput(source="/opt/ml/processing/evaluation", output_name="evaluation")],
        code="pipelines/abalone/evaluate_script.py",
        property_files=[evaluation_report],
    )

    condition = ConditionLessThanOrEqualTo(
        left= auc_threshold,
        right= JsonGet(step=step_evaluate, property_file=evaluation_report, json_path="regression_metrics.auc.value"),
    )
    

    
# Tạo model từ model artifacts
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=step_evaluate.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
            content_type="application/json"
            )
    )
    
    
    register_model_step = RegisterModel(
        name="RegisterModelStep",
        model=Model(
            image_uri="600627364468.dkr.ecr.us-east-1.amazonaws.com/mq/inference:latest",
            model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,  # Đợi train xong mới có giá trị
            role=role,
            sagemaker_session=sagemaker_session
        ),
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )

    step_register_condition = ConditionStep(name="CheckModelQuality", conditions=[condition], if_steps=[register_model_step], else_steps=[], depends_on=[step_evaluate])
    

    

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
            auc_threshold,
        ],
        steps=[step_process, step_train, step_evaluate, step_register_condition],
        sagemaker_session=sagemaker_session,
    )
    return pipeline
