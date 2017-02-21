import os, sys, cPickle as pkl
from collections import OrderedDict as ordict
import numpy as np, tensorflow as tf
import datasets

def main():
  datasource = datasets.MscocoNP(data_dir="/Tmp/cooijmat/mscoco")
  
  tfrecord_dir = os.environ["MSCOCO_TFRECORD_DIR"]
  tf.gfile.MakeDirs(tfrecord_dir)

  alphabet_path = os.path.join(tfrecord_dir, "alphabet.pkl")
  print "writing to", alphabet_path
  pkl.dump(datasource.alphabet, open(alphabet_path, "wb"))
  print "done"

  for fold in "train valid".split():
    output_path = os.path.join(tfrecord_dir, fold + ".tfrecords")
    print "writing to", output_path
    writer = tf.python_io.TFRecordWriter(output_path)
    for filename in datasource.get_filenames(fold):
      identifier = os.path.splitext(os.path.basename(filename))[0]

      caption_strings = caption_dict[identifier]
      np.random.shuffle(caption_strings)
      caption_string = "|".join(caption_strings)
      caption = bytes(bytearray([alphabet[c] for c in caption_string]))

      image = open(filename, "rb").read()

      def zzz(x):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[x]))

      example = tf.train.Example(features=tf.train.Features(
        feature=dict(image=zzz(image),
                     caption=zzz(caption),
                     identifier=zzz(identifier)))
      writer.write(example.SerializeToString())
    writer.close()

if __name__ == "__main__":
  main()
