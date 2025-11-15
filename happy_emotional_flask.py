####################################################################################################
# Do not modify this code block #
# Use legacy Keras for compatibility with transformers
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

# Import monitoring packages
import logging
import traceback
import json

# Import flask packages
import werkzeug

from werkzeug.wrappers import Response
#pip  install --upgrade Flask==2.2.5
from flask import Flask, make_response, request, Response

from flask_restx import Api, Resource
from werkzeug.utils import cached_property
####################################################################################################
 #pip  install --upgrade Werkzeug==2.2.3
# Custom packages
import numpy as np
import sys
import tensorflow as tf
from logging.handlers import RotatingFileHandler
import io
from hsjeon_datascience_nlp import create_sentiment_bert, mean_answer_label, KoBertTokenizer


# Use CPU for local testing, GPU in production
import os
if os.path.exists('/app'):  # Running in Docker
    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"], cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
else:  # Running locally
    strategy = tf.distribute.get_strategy()  # Default strategy (CPU)

# Model loading configuration
# Set USE_LOCAL_MODEL=True to use local file, False to use HuggingFace (default)
USE_LOCAL_MODEL = os.environ.get('USE_LOCAL_MODEL', 'False').lower() == 'true'
LOCAL_MODEL_PATH = './data/fsi_comment_sentiment_model.h5'
HUGGINGFACE_MODEL_REPO = 'HyeonSang/kobert-sentiment'

if USE_LOCAL_MODEL:
    # Load from local file
    print(f"Loading model from local file: {LOCAL_MODEL_PATH}")
    with strategy.scope():
        sentiment_model = create_sentiment_bert(learning_rate=0.000001, SEQ_LEN=32, DROPOUT=0.01, OUTPUT_CNT=1, loss_type='BinaryCrossentropy')
        sentiment_model.load_weights(LOCAL_MODEL_PATH)
        print("Local model loaded successfully")
else:
    # Load from HuggingFace Hub (default)
    print(f"Loading model from HuggingFace Hub: {HUGGINGFACE_MODEL_REPO}")
    from huggingface_hub import hf_hub_download
    import tensorflow as tf
    model_file = hf_hub_download(repo_id=HUGGINGFACE_MODEL_REPO, filename="tf_model.h5")
    with strategy.scope():
        sentiment_model = create_sentiment_bert(learning_rate=0.000001, SEQ_LEN=32, DROPOUT=0.01, OUTPUT_CNT=1, loss_type='BinaryCrossentropy')
        # Try load_weights with by_name and skip_mismatch
        try:
            sentiment_model.load_weights(model_file, by_name=True, skip_mismatch=True)
            print("HuggingFace model loaded successfully (with by_name=True)")
        except Exception as e:
            print(f"Failed to load with by_name, trying direct load_model: {e}")
            # If that fails, try loading the full model
            sentiment_model = tf.keras.models.load_model(model_file, compile=False)
            print("HuggingFace model loaded successfully (load_model)")

# Initialize tokenizer using huggingface_hub
print("Initializing tokenizer...")
from huggingface_hub import hf_hub_download
vocab_file = hf_hub_download(repo_id="monologg/kobert", filename="tokenizer_78b3253a26.model")
vocab_txt = hf_hub_download(repo_id="monologg/kobert", filename="vocab.txt")
tokenizer = KoBertTokenizer(vocab_file=vocab_file, vocab_txt=vocab_txt)
print(f"Tokenizer type: {type(tokenizer)}")
print(f"Tokenizer initialized successfully")

mod = sys.modules[__name__]


def sentence_convert_data(data):
    SEQ_LEN = 32
    tokens, masks, segments = [], [], []
    token = tokenizer.encode(data, max_length=SEQ_LEN, padding='max_length', truncation=True)

    num_zeros = token.count(0)
    mask = [1] * (SEQ_LEN - num_zeros) + [0] * num_zeros
    segment = [0] * SEQ_LEN

    tokens.append(token)
    segments.append(segment)
    masks.append(mask)

    tokens = np.array(tokens)
    masks = np.array(masks)
    segments = np.array(segments)

    return [tokens, masks, segments]

# Flask app
app = Flask(__name__)
api = Api(app, version='1.0', title='Welcome to the AWS FSI GameDay AIML', doc='/', description='NLP 감정분석 API')
app.config.SWAGGER_UI_DOC_EXPANSION = 'list'
ns_conf = api.namespace('Korean sentiment analysis', description='한국어 문장 감정 분석')

input_parser = ns_conf.parser()
input_parser.add_argument('input_data', type=str, help='input data', location='form')

logFormatStr = '%(asctime)s | p%(process)s | %(levelname)s | %(message)s |'
formatter = logging.Formatter( logFormatStr)
handler = RotatingFileHandler('./log_data.log', maxBytes=10000000, backupCount=5,encoding = 'utf-8')
handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)

#log = logging.getLogger('werkzeug')
log = logging.getLogger('werkzeug')
log.setLevel(logging.INFO)
log.addHandler(handler)
#sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
#sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')


# Custom API class
@ns_conf.route("", methods=['POST'])
#@ns_conf.response(404, "can not find input_data value")
@ns_conf.param("input_data", "input your data")
class ModelApi(Resource):
    @staticmethod
    @ns_conf.expect(input_parser)
    def post():
        # custom variables and function
        input_data = input_parser.parse_args().pop('input_data')
        if not input_data:
            input_data = json.loads(request.get_data(), encoding='utf-8')['input_data']

        cat_dict = {'0': "부정적인 문장", '1': "긍정적인 문장"}

        data_x = sentence_convert_data(input_data)

        with strategy.scope():
            predict = sentiment_model.predict(data_x)
            predict_value = np.ravel(predict)
            predict_answer = np.round(predict_value,0).item()

        if predict_answer == 0:
            result_txt = "부정적인 문장입니다."  +" 부정 확률 : [ " + str("%0.2f" %((1-predict_value)*100))+"% ]"
            print("부정적인 문장입니다. : (부정 확률 : %.2f) " % (1-predict_value))
        elif predict_answer == 1:
            result_txt = "긍정적인 문장입니다."  +" 긍정 확률 : [ " + str("%0.2f" %(predict_value*100))+"% ]"
            print("긍정적인 문장입니다. : (긍정 확률 : %.2f) " % predict_value)

        print("pred : ",predict_answer)

        log.debug("JHS TEST LOG :"+ input_data)

        res = Response(json.dumps(result_txt, ensure_ascii=False).encode('utf8'), content_type='application/json; charset=utf-8')
        # res.headers["Access-Control-Allow-Origin"] = "*"
        # print(res.headers)


        return res


# Logging

#Error handling
@app.errorhandler(404)
def not_found(message):
    print(f"404 Error: {message}")
    return make_response(message, 404)


@app.errorhandler(400)
def bad_request(message):
    print(f"400 Error: {message}")
    return make_response(message, 400)


@app.errorhandler(Exception)
def internal_error(error):
    error_traceback = traceback.format_exc()
    print("=" * 80)
    print("ERROR OCCURRED:")
    print(error_traceback)
    print("=" * 80)
    return make_response(error_traceback, 500)


# Run api (main)
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

