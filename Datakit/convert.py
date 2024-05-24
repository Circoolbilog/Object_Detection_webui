"""
The Convert class  is a collection of functions which are commonly used in parsing and
converting data to different formats.

"""
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


class Parser:
    def __init__(
            self,
            base_directory=os.getcwd(),
            data_rel_directory=None,
            data_direct_directory=None,
            data_set=None,
            all_gen_dir=None,
            mkdir=None,
            find=None,
            found="Found FILES: \n"
    ):

        self.data_direct_directory=data_direct_directory
        self.base_directory = base_directory
        self.data_rel_directory = data_rel_directory
        self.data_set = data_set
        self.all_gen_dir = all_gen_dir
        self.mkdir = mkdir
        self.find = find
        self.found = found
        print(self.base_directory)

    def parse_xml(self):
        to_df = []
        fields = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
        # file_location = '{}/{}'.format(self.base_directory, self.data_rel_directory)
        # file_location = os.path.join(self.base_directory, self.data_rel_directory)
        file_location = self.data_direct_directory
        list_all_files = glob.glob('{}/*.xml'.format(file_location))
        # list_all_files = glob.glob('{}\\*.xml'.format(file_location))

        # parse data
        for xml_file in list_all_files:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            filename = '{}/{}'.format(file_location, root.find('filename').text)
            # filename = '{}\\{}'.format(file_location, root.find('filename').text)
            print(filename)
            width = int(root.find('size')[0].text)
            height = int(root.find('size')[1].text)
            print("Parsing ", filename)

            for member in root.findall('object'):
                obj_class = member[0].text
                xmin = int(member[4][0].text)
                ymin = int(member[4][1].text)
                xmax = int(member[4][2].text)
                ymax = int(member[4][3].text)
                value = (filename, width, height, obj_class, xmin, ymin, xmax, ymax)
                if (os.path.exists(filename)):
                    to_df.append(value)
                if obj_class == self.find:
                    self.found=self.found +filename+"\n"

        xml_df = pd.DataFrame(to_df, columns=fields)
        return xml_df


class Converter(Parser):
    def find_annotation(self):
        xml_df = self.parse_xml()
        print(self.found)

    def xml_write_to_csv(self):
        xml_df = self.parse_xml()
        # csv_filename='{}/{}/{}_labels.csv'.format(self.base_directory, self.all_gen_dir, self.data_set)
        csv_filename = os.path.join(self.base_directory, self.all_gen_dir, self.data_set)
        csv_file_create = open(csv_filename, 'w')
        xml_df.to_csv(csv_filename, index=None)
        print("Success! ", csv_filename)
        csv_file_create.close()

    def yolo_write_to_pascalVOC_xml(self):
        print(self.f)


