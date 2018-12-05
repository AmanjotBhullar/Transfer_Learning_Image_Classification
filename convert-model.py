import tensorflow as tf
from tensorflow.contrib import layers

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils as saved_model_utils
import tensorflow.python.saved_model.simple_save


export_dir = 'religion-converted-model2'
retrained_graph = '/Users/vishalprabhachandar/Documents/Delve/Model-religion/retrained_graph.pb'
label_count = 7

class Model(object):
    def __init__(self, label_count):
        self.label_count = label_count

    def build_prediction_graph(self, g):
        inputs = {
            'key': keys_placeholder,
            'image_bytes': tensors.input_jpeg
        }

        keys = tf.identity(keys_placeholder)
        outputs = {
            'key': keys,
            'prediction': g.get_tensor_by_name('final_result:0')
        }

        return inputs, outputs

    def export(self, output_dir):
        with tf.Session(graph=tf.Graph()) as sess:
            # This will be our input that accepts a batch of inputs
            image_bytes = tf.placeholder(tf.string, name='input', shape=(None,))
            # Force it to be a single input; will raise an error if we send a batch.
            coerced = tf.squeeze(image_bytes)
            # When we import the graph, we'll connect `coerced` to `DecodeJPGInput:0`
            input_map = {'DecodeJpeg/contents:0': coerced}

            with tf.gfile.GFile(retrained_graph, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, input_map=input_map, name="")

            keys_placeholder = tf.placeholder(tf.string, shape=[None])

            inputs = {'image_bytes': image_bytes, 'key': keys_placeholder}

            keys = tf.identity(keys_placeholder)
            outputs = {
                'key': keys,
                'prediction': tf.get_default_graph().get_tensor_by_name('final_result:0')}

            tf.saved_model.simple_save(sess, output_dir, inputs, outputs)

model = Model(label_count)
model.export(export_dir)
