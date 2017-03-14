import os, sys, cPickle as pkl
from collections import OrderedDict as ordict
import numpy as np, tensorflow as tf
import datasets
from holster import H

def _int64_feature(value): return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value): return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))
def _int64_feature_list(values): return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])
def _bytes_feature_list(values): return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])

def main():
  datasource = datasets.MscocoNP(H(data_dir="/Tmp/cooijmat/mscoco", hp=H(caption=H(token="character"))))

  tfrecord_dir = os.environ["MSCOCO_TFRECORD_DIR"]
  tf.gfile.MakeDirs(tfrecord_dir)

  tokenizers = ordict((token, datasource.get_tokenizer(token))
                      for token in "character word".split())

  def _to_sequence_example(image_path):
    identifier = os.path.splitext(os.path.basename(image_path))[0]
    caption_words = tokenizers["word"].process(datasource.get_caption_string(identifier))
    caption_characters = tokenizers["character"].process(datasource.get_caption_string(identifier))
    with tf.gfile.FastGFile(image_path, "rb") as f:
      jpeg = f.read()
    return tf.train.SequenceExample(
      context=tf.train.Features(feature={
        "image/identifier": _bytes_feature(identifier),
        "image/data": _bytes_feature(jpeg),
       }),
      feature_lists=tf.train.FeatureLists(feature_list={
        "image/caption_characters": _int64_feature_list(caption_characters),
        "image/caption_words": _int64_feature_list(caption_words),
       }))

  for fold in "train valid".split():
    output_path = os.path.join(tfrecord_dir, fold + ".tfrecords")
    print "writing to", output_path
    writer = tf.python_io.TFRecordWriter(output_path)
    for filename in datasource.get_filenames(fold):
      example = _to_sequence_example(filename)
      writer.write(example.SerializeToString())
    writer.close()

if __name__ == "__main__":
  main()
