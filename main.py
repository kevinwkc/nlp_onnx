import os
import pandas as pd
import numpy as np
from transformers import RobertaTokenizer
import torch

import onnxruntime
import onnx

import mlflow
import mlflow.pyfunc

my_run_id=None
classification_model = 'roberta-sequence-classification-9.onnx'
    #'/d/temp/models/text/machine_comprehension/roberta/production/app_vol/roberta-sequence-classification-9.onnx'
    #'/data/roberta-sequence-classification-9.onnx'

class BertWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        from transformers import RobertaTokenizer

        # load tokenizer and model from artifacts in model context
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', cache_dir="/tmp")


    def predict(self, context, model_input):
        import pandas as pd
        import torch
        answers = []
        for _, row in model_input.iterrows():

            txt = row["txt"]

            input_ids = torch.tensor(self.tokenizer.encode(txt, add_special_tokens=True)).unsqueeze(0)  # Batch size 1

            ort_session = onnxruntime.InferenceSession(classification_model)

            def to_numpy(tensor):
                return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_ids)}
            ort_out = ort_session.run(None, ort_inputs)

            pred = np.argmax(ort_out)
            answer=None
            if (pred == 0):
                answer="negative"
            elif (pred == 1):
                answer="positive"

            answers.append(answer)
        return pd.Series(answers)


if __name__ == "__main__":
    model = onnx.load(classification_model)
    with mlflow.start_run() as run:
        mlflow.set_tag("model_flavor", "pytorch")
        #mlflow.onnx.save_model(model, path='roberta-saved', conda_env="conda.yaml")
        mlflow.log_metric('accuracy', 0.99)
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base', cache_dir='/tmp')
        mlflow.pyfunc.log_model(artifact_path='model', python_model=BertWrapper(), registered_model_name="mybert") #, artifacts=artifacts

        my_run_id=mlflow.active_run().info.run_id

model_uri=f'runs:/{my_run_id}/model'
sentence_classifier=mlflow.pyfunc.load_model(model_uri=model_uri)