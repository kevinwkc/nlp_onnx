#!/usr/bin/env python

import torch
import numpy as np
import onnxruntime
#import confuse

'''
API LIBRARY
'''
from flask import Flask, request
from waitress import serve

app = Flask(__name__)

'''
SENTIMENT LIBRARY
'''
# from simpletransformers.model import TransformerModel
# from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained('roberta-base', cache_dir="cache/")
pos = "My NLP skills is not bad"
neg = "This unethical person behavior is not really good"
txt=neg

# MULTI LANGUAGE ENTITY RECOGNITION
# https://github.com/deepmipt/DeepPavlov/blob/0.14.0/deeppavlov/configs/ner/ner_ontonotes_bert_mult.json
# https://github.com/deepmipt/DeepPavlov/issues/1097
#
# Could u provide a link of the dataset?
#     http://files.deeppavlov.ai/deeppavlov_data/ontonotes_ner.tar.gz
#
# 2021-01-17 15:33:09.487 INFO in deeppavlov.core.data.utils[utils] at line 94: Downloading from http://files.deeppavlov.ai/deeppavlov_data/bert/multi_cased_L-12_H-768_A-12.zip to C:\Users\kevin\.deeppavlov\downloads\multi_cased_L-12_H-768_A-12.zip


# from deeppavlov import configs, build_model
# import json
# import tensorflow
# #http://docs.deeppavlov.ai/en/master/intro/configuration.html
# with configs.ner.ner_ontonotes_bert_mult.open(encoding='utf8') as f:
#     ner_config = json.load(f)
# ner_config['metadata']['variables']['ROOT_PATH'] = './'
# ner_config['metadata']['download'] = [ner_config['metadata']['download'][-1]]  # do not download the pretrained ontonotes model
#
# ner_model = build_model(ner_config)
# # ner_model = build_model(configs.ner.ner_ontonotes_bert_mult, download=True)
# ner_model(['Bob Ross lived in Florida'])
#
# '''
#      GET
#          localhost: 8000 / v1 / erecog
#          {
#              "txt": "KINTON RAMEN TORONTO ON -$18.58"}
#
#      RETURN positive
# '''
#
#
# @app.route('/v1/erecog', methods=['GET'])
# def erecog():
#     content = request.json
#     txt = content['txt']
#     return ner_model([txt])

'''
     GET
         localhost: 8000 / v1 / sentiment
         {
             "txt": "Best quality  I've tasted in all you can eat restaurant  . I like that beef tongues are actually tasty. Seafood dishes was fresh. The brisket beef is super yummyalso. The cool as a Kurobuta sausage is my favorite dish here. We always go this branch because staff are very nice and attentive. Prices are adequate and reasonable for 45 aprox. per person btw There is 5% when you pay with cash !!"}
             
     RETURN positive
'''
@app.route('/v1/sentiment', methods=['GET'])
def sentiment():
    content = request.json
    txt = content['txt']

    input_ids = torch.tensor(tokenizer.encode(txt, add_special_tokens=True)).unsqueeze(0)  # Batch size 1

    ort_session = onnxruntime.InferenceSession("/data/roberta-sequence-classification-9.onnx")

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_ids)}
    ort_out = ort_session.run(None, ort_inputs)

    pred = np.argmax(ort_out)
    if(pred == 0):
        return "negative"
    elif(pred == 1):
        return "positive"

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080)



