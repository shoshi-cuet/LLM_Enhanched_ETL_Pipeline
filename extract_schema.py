import re
import json
import xml.etree.ElementTree as ET

import pandas as pd

from helper import Helper

def xml_to_dict(xml):
    '''
    This is a recursive function to convert XML tree into a dictionary
    '''
    if len(xml) == 0:  # Check if the element has no children
        return xml.text
    else:
        dict_data = {}
        for child in xml:
            child_data = xml_to_dict(child)
            if child.tag in dict_data:  # Check if the tag already exists
                if isinstance(dict_data[child.tag], list):
                    dict_data[child.tag].append(child_data)
                else:
                    dict_data[child.tag] = [dict_data[child.tag], child_data]
            else:
                dict_data[child.tag] = child_data
        return dict_data

# file path for specific databases
database = 'northwind'
test_file_name = 'products'
model_name = 'llama2'
file_path = f'./data/output/{database}/{test_file_name}_schema_generated_{model_name}.csv'

# extracts the content into a dataframe
extracted_content = Helper.extract_xml_schema(file_path)
extracted_df = pd.DataFrame(extracted_content, columns=['schema'])

# patterns are matched and the most common pattern is generated
pattern_counts = extracted_df['schema'].value_counts()
most_common_pattern = pattern_counts.idxmax()
most_common_pattern

root = ET.fromstring(most_common_pattern)
    
# converts XML to a dictionary
xml_dict = xml_to_dict(root)

# converts the dictionary to JSON
json_string = json.dumps(xml_dict, indent=4)

with open("./output.json", "w") as file:
    file.write(json_string)
