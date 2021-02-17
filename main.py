import os
import pandas as pd
import numpy as np
from transformers import RobertaTokenizer
import torch

import onnxruntime
import onnx

import mlflow
import mlflow.pyfunc

import requests
import yaml
from tqdm import tqdm

import tempfile

my_run_id=None

tracking_uri="http://ryzen:5000/"
mlflow.set_tracking_uri(tracking_uri)
mlflowClient = mlflow.tracking.MlflowClient(tracking_uri)

os.environ['MLFLOW_TRACKING_URI'] = tracking_uri
os.environ['MLFLOW_ARTIFACT_URI'] = tracking_uri

my_artifact_path = "model"
saved_path = 'roberta-saved'
classification_model = 'roberta-sequence-classification-9.onnx'
    #'/d/temp/models/text/machine_comprehension/roberta/production/app_vol/roberta-sequence-classification-9.onnx'
    #'/data/roberta-sequence-classification-9.onnx'
data_path = 'data'


TEMPLATE_BASE = "https://github.com/onnx/models/raw/{}/text/machine_comprehension/roberta/model/roberta-base-11.onnx"
TEMPLATE_MODEL = "https://github.com/onnx/models/raw/{}/text/machine_comprehension/roberta/model/roberta-sequence-classification-9.onnx"




def download(url, fname):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
            desc=fname,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)



class BertWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        from transformers import RobertaTokenizer

        # load tokenizer and model from artifacts in model context
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', cache_dir="/tmp")


    def predict(self, context, model_input):
        import pandas as pd
        import torch
        df = pd.DataFrame()
        # print(context)
        print(dir(context.artifacts))
        print(context.artifacts)
        print(context.artifacts["bert_model"])
        for _, row in model_input.iterrows():

            txt = row["txt"]

            input_ids = torch.tensor(self.tokenizer.encode(txt, add_special_tokens=True)).unsqueeze(0)  # Batch size 1

            ort_session = onnxruntime.InferenceSession(classification_model)

            def to_numpy(tensor):
                return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_ids)}
            ort_out = ort_session.run(None, ort_inputs)

            pred = np.argmax(ort_out)

            if (pred == 0):
                df = df.append({'txt': txt, 'sentiment': 'negative'}, ignore_index=True)
            elif (pred == 1):
                df = df.append({'txt': txt, 'sentiment': 'positive'}, ignore_index=True)

        return df


if __name__ == "__main__":
    model = onnx.load(classification_model)
    #tempfile.TemporaryDirectory() as tmp_dir,
    with mlflow.start_run(run_name='RoBERTa') as run:
        # Added this line
        print('tracking uri:', mlflow.get_tracking_uri())
        print('artifact uri:', mlflow.get_artifact_uri())

        mlflow.set_tag("model_flavor", "pytorch")
        # my_artifacts = {
        #     "bert_model": os.path.join(str(tmp_dir), saved_path, "model", classification_model),
        #     "bert_model_base_tokenizer": os.path.join(str(tmp_dir), saved_path, "base", "roberta-base-11.onnx"),
        #     "metadata": os.path.join(str(tmp_dir), data_path, "metadata.yaml")
        # }
        my_artifacts = {
            "bert_model": os.path.join(os.getcwd(), saved_path, "model", classification_model),
            "bert_model_base_tokenizer": os.path.join(os.getcwd(), saved_path, "base", "roberta-base-11.onnx"),
            "metadata": os.path.join(os.getcwd(), data_path, "metadata.yaml")
        }

        #https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.save_model
        if(not os.path.exists(saved_path)):
            mlflow.onnx.save_model(model, path=saved_path, conda_env="conda.yaml")

        mlflow.log_metric('accuracy', 0.99)
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base', cache_dir='/tmp')

        #https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.log_model
        print(my_artifacts)


        if not os.path.exists(data_path):
            os.makedirs(data_path)
        upstream_roberta_base_version='841d321'
        metadata = {'bert_model_base_tokenizer_revision': upstream_roberta_base_version}
        with open(my_artifacts['metadata'], "w") as f:
            yaml.safe_dump(metadata, stream=f)

        # Download the base tokenizer and model

        roberta_base_version='841d321fa51acc648d605a33c87ebe6726da8153'
        roberta_model_version='bf6e73c4c68db02dc9cecd631a4a03a453932de0'
        base_url = TEMPLATE_BASE.format(roberta_base_version)
        model_url = TEMPLATE_MODEL.format(roberta_model_version)
        # download(base_url, my_artifacts['bert_model_base_tokenizer'])
        # download(model_url, my_artifacts['bert_model'])

        # mlflow.log_artifact("roberta-sequence-classification-9.onnx")
        mlflow.log_artifact(my_artifacts['bert_model_base_tokenizer'])

        # my_artifact_path=mlflow.get_artifact_uri()
        mlflow.pyfunc.log_model(artifact_path=my_artifact_path, python_model=BertWrapper(), registered_model_name="mybert",  conda_env="conda.yaml") #, artifacts=my_artifacts)

        from mlflow.tracking.artifact_utils import get_artifact_uri


        my_run_id=mlflow.active_run().info.run_id
        source = get_artifact_uri(run_id=my_run_id, artifact_path=my_artifact_path)


