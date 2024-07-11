import numpy as np
import pandas as pd
import re
import evaluate
from sklearn.model_selection import train_test_split

class Helper():
    def remove_commas(file_path):
        '''
        This function removes unformatted commas from the .csv files in the preprocessing step.
        '''
        with open(file_path, 'r') as f:
            lines = f.readlines()
    
        updated_lines = [line.replace(',', ' ') for line in lines]

        with open(file_path, 'w') as f:
            f.writelines(updated_lines)


    def stratify_adventure(file_path):
        '''
        This function creates 10% stratified adventureworks train and validation files without class imbalance.

        Args:
            file_path: train or validation file path of adventureworks.
        Return:
            saved sampled stratified .csv file
        '''
        # Read the data and prepare for stratification
        df = pd.read_csv('/path/to/adventureworks/train/and/validation/files')
        X = df.drop('response', axis=1)
        y = df['response']

        # Stratified sampling to select and save
        X_sample, _, y_sample, _ = train_test_split(X, y, test_size=0.9, stratify=y, random_state=42)
        sampled_df = pd.concat([X_sample, y_sample], axis=1)
        sampled_df.to_csv('path/to/output/files')


    def formatting_test_prompts_func(example):
        '''
        This function helps to create the prompt template to pass in model generation without specifying the output. Used for the testing purpose.
        '''
        output_texts = []

        for i in range(len(example['prompt'])):
            if example['prompt'][i] == 'prompt':
                pass
            else:
                text = f"### Input: ```{example['prompt'][i]}```\n ### Output: ''"
                output_texts.append(text)

        return output_texts
    
    
    def formatting_prompts_func(example):
        '''
        This function helps to create the prompt template to pass in formatting_func in model trainer specifying both input and output. 
        '''
        output_texts = []

        for i in range(len(example['prompt'])):

            text = f"### Input: ```{example['prompt'][i]}```\n ### Output: {example['response'][i]}"

            output_texts.append(text)

        return output_texts
    

    def extract_xml_schema(file_path):
        '''
        This function extracts only the part of the generated output included between <root> and </root>
        '''
        with open(file_path, 'r') as file:
            content = file.read()
    
        matches = re.findall(r'<root>.*?</root>', content, re.DOTALL) # <root>...</root> inclusive output
        return matches
    
def main():
    Helper()

if __name__ =='__main__':
    main()