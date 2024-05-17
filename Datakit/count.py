import os
import pandas as pd

class Counter():
	def __init__(
		self,
		base_directory = os.getcwd(),
		csv_filename = None,
		txt_filename = None,
		pbtxt_filename = None,
		mkdir = None
		):

		self.base_directory = base_directory
		self.csv_filename = csv_filename
		self.unique_dataframe = None
		self.n_count = None
		self.mkdir = mkdir
		self.txt_filename = txt_filename
		self.pbtxt_filename = pbtxt_filename
		self.filename = os.path.join(self.base_directory, self.csv_filename)
		self.df = pd.read_csv(self.filename)

	def unique_n(self):
		df = (self.df).sort_values(by='class')
		unique_n = df.drop_duplicates(subset=['class'])
		unique_n =unique_n['class'].tolist()
		print(unique_n)
		return unique_n
		
	def count_n(self):
		df = (self.df).sort_values(by='class')
		n_count = df['class'].value_counts()
		print(n_count)
		return n_count

	def write_labelmap_txt(self):
		input_list=self.unique_n()
		txt_file = os.path.join(self.base_directory, self.txt_filename)

		if os.path.exists(txt_file):
			os.remove(txt_file)

		for item in input_list:
			with open(txt_file, 'a') as f:
				f.write('{}\n'.format(item))
		print("Success {}!".format(txt_file))

	def write_labelmap_pbtxt(self):
		pbtxt_file = os.path.join(self.base_directory, self.pbtxt_filename)


		if os.path.exists(pbtxt_file):
			os.remove(pbtxt_file)

		for i in range(len(self.unique_n())):
			with open(pbtxt_file, 'a') as f:
				context = "item" + "{\n" + "	id: " + "{}\n".format(i+1) + "	name: '{}'".format(self.unique_n()[i]) + "\n}\n"
				f.write(context)

		print("Success {}!".format(pbtxt_file))