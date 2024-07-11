import pandas as pd
import re
import xml.etree.ElementTree as ET

#from helper import Helper

# Function to extract content including <root> and </root>
def extract_root_content(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Find all matches of <root>...</root>, including the tags
    matches = re.findall(r'<root>.*?</root>', content, re.DOTALL)
    return matches


# Function to parse XML and extract schema details
def parse_schema(xml_content):
    
    root = ET.fromstring(xml_content)
    schema = {}
    for child in root:
        field_name = child.tag
        field_details = {}
        for sub_child in child:
            tag_name = sub_child.tag
            tag_value = sub_child.text
            if tag_name == 'constraints':
                if 'constraints' not in field_details:
                    field_details['constraints'] = []
                field_details['constraints'].append(tag_value)
            else:
                field_details[tag_name] = tag_value
        schema[field_name] = field_details
    return schema
    
# Function to compare two schemas and calculate accuracy
def compare_schemas(predicted_schema, actual_schema):
    '''
        Calculates the accuracy of the predicted schema structure. 
        Accuracy is calculated by dividing the number of correct elements by the total number of elements checked.

        Args:
            predicted: predicted schema structure.
            actual: actual schema structure
        Returns:
            accuracy: calculated accuracy of the predicted schema structure.
    '''
    correct_fields = 0
    total_fields = len(actual_schema)

    for field in actual_schema:
        if field in predicted_schema:
            actual_constraints = set(actual_schema[field].get('constraints', []))
            predicted_constraints = set(predicted_schema[field].get('constraints', []))

            if (actual_schema[field]['type'] == predicted_schema[field]['type'] and
                actual_constraints == predicted_constraints):
                correct_fields += 1

    schema_accuracy = correct_fields / total_fields if total_fields > 0 else 0
    return schema_accuracy
    
def calc_accuracy(actual_file_path, predict_file_path):
    # Create a DataFrame with the extracted content
    extracted_df = pd.DataFrame(extract_root_content(predict_file_path), columns=['schema'])
    predicted_len = len(extracted_df)

    pattern_counts = extracted_df['schema'].value_counts()
    predicted_schema_struc = pattern_counts.idxmax()

    actual_df = pd.read_csv(actual_file_path)
    actual_len = len(actual_df)
    actual_schema_struc = actual_df['response'][0]

    # Parse the XML contents
    predicted_schema = parse_schema(predicted_schema_struc)
    actual_schema = parse_schema(actual_schema_struc)

    # Compare schemas and calculate accuracy
    schema_accuracy = compare_schemas(predicted_schema, actual_schema)

    schema_detection_acc = predicted_len / actual_len
    accuracy = (schema_detection_acc + schema_accuracy) / 2

    return accuracy

def main():
    # Specify the path to your file
    predict_file_path = '/Users/zohora/Desktop/thesis/implementation/data/output/northwind/products_schema_generated_llama2.csv'
    actual_file_path = 'data/input/NorthWind/processed/xml_output/window_4/test/products.csv'

    accuracy = calc_accuracy(actual_file_path, predict_file_path)
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__=="__main__":
    main()