from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow.compat.v1 as tf

from PIL import Image
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from collections import namedtuple, OrderedDict

import contextlib2
from object_detection.dataset_tools import tf_record_creation_util


class shardTFRecord():
	def __init__(self, 
		base_directory=os.getcwd(),
		csv_file_input=None,
		pbtxt_labels=None,
		data_type=None,
		all_gen_dir=None,
		num_of_shards=1):

		self.base_directory = base_directory
		self.csv_file_input = csv_file_input
		self.pbtxt_labels = pbtxt_labels
		self.data_type = data_type
		self.all_gen_dir = all_gen_dir
		self.num_of_shards = num_of_shards

		self.output_filebase=os.path.join(self.base_directory, 'all_generated', '{}.tfrecord'.format(self.data_type))
		self.labelmap_dir = os.path.join(self.base_directory, self.pbtxt_labels)
		self.labelmap_dict = label_map_util.get_label_map_dict(self.labelmap_dir)
		self.csv_filename = os.path.join(self.base_directory, self.csv_file_input)
		self.group = None
		self.dfsamples = pd.read_csv(self.csv_filename)

	def split(self):
		data = namedtuple('data', ['filename', 'object'])
		gb=(self.dfsamples).groupby('filename')
		return [data(filename, gb.get_group(x)) for filename , x in zip(gb.groups.keys(), gb.groups)]

	def create_tf_example(self):
		with tf.gfile.GFile(self.group.filename, 'rb') as fid:
			encoded_jpg=fid.read()

		encoded_jpg_io = io.BytesIO(encoded_jpg)
		image = Image.open(encoded_jpg_io)
		width, height = image.size

		filename = self.group.filename.encode('utf8')
		image_format = b'jpg'
		xmins=[]
		xmaxs=[]
		ymins=[]
		ymaxs=[]
		classes_text=[]
		classes=[]

		for index, row in self.group.object.iterrows():
			xmins.append(float(row['xmin']/width))
			xmaxs.append(float(row['xmax']/width))
			ymins.append(float(row['ymin']/height))
			ymaxs.append(float(row['ymax'])/height)
			classes_text.append(row['class'].encode('utf8'))
			classes.append(self.labelmap_dict[row['class']])

		tf_example = tf.train.Example(features=tf.train.Features(feature={
			'image/height':dataset_util.int64_feature(height),
			'image/width' :dataset_util.int64_feature(width),
			'image/filename' :dataset_util.bytes_feature(filename),
			'image/source_id' :dataset_util.bytes_feature(filename),
			'image/encoded' :dataset_util.bytes_feature(encoded_jpg),
			'image_format' :dataset_util.bytes_feature(image_format),
			'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
			'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
			'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
			'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
			'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
			'image/object/class/label': dataset_util.int64_list_feature(classes),
			}))
		return tf_example


	def convert(self):
		dfsamples=self.dfsamples
		grouped = self.split()

		with contextlib2.ExitStack() as tf_record_close_stack:
			output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
					tf_record_close_stack,  self.output_filebase, self.num_of_shards
				)
			for group in grouped:
				self.group = group
				index = grouped.index(group)
				labelmap_dict = self.labelmap_dict
				tf_example = self.create_tf_example()
				output_shard_index = index%int(self.num_of_shards)
				output_tfrecords[output_shard_index].write(tf_example.SerializeToString())
			output_tfrecords[output_shard_index].close()
			print("Success {}".format(output_shard_index))
