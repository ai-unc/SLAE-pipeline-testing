import sys
import os
import re
import json
from PyPDF2 import PdfReader
from typing import List, Tuple
#import fitz

def extract_sticky_notes(pdf_path):
    sticky_notes = []
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            if '/Annots' in page:
                annotations = page['/Annots']
                for annot in annotations:
                    annotation_object = annot.get_object()
                    if '/Contents' in annotation_object:
                        contents = annotation_object['/Contents']
                        variable_one_name = re.search(r'variable 1: (.+?)\n', contents, re.IGNORECASE)
                        variable_two_name = re.search(r'variable 2: (.+?)\n', contents, re.IGNORECASE)
                        relationship_classification = re.search(r'relationship: (.+?)(?:\n|$)', contents, re.IGNORECASE)
                        if relationship_classification:
                            relationship_value = relationship_classification.group(1)
                        else:
                            relationship_classification2 = re.search(r'relationship classification: (.+?)(?:\n|$)', contents, re.IGNORECASE)
                            relationship_value = relationship_classification2.group(1) if relationship_classification2 else ""
                        is_causal = re.search(r'is causal: (.+?)\n', contents, re.IGNORECASE)
                        attributes = re.search(r'attributes: (.+?)(?:\n|$)', contents, re.IGNORECASE)
                        supporting_text = re.search(r'text: (.+?)(?:\n|$)', contents, re.IGNORECASE)
                        sticky_notes.append({
                            'VariableOneName': variable_one_name.group(1) if variable_one_name else "",
                            'VariableTwoName': variable_two_name.group(1) if variable_two_name else "",
                            'RelationshipClassification': relationship_value,
                            'IsCausal': is_causal.group(1) if is_causal else "",
                            'Attributes': attributes.group(1) if attributes else "",
                            'SupportingText': supporting_text.group(1) if supporting_text else "",
                        })
    return sticky_notes


def extract_pdf_metadata(pdf_path):
   with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        metadata = pdf_reader.metadata
        page0 = pdf_reader.pages[0]
        text = page0.extract_text()
        doi_pattern1 = re.search(r'doi:\s*(\S+)', text, re.IGNORECASE)
        doi_pattern2 = re.search(r'http://\S+\.doi\.\S+', text, re.IGNORECASE)
        doi_pattern3 = re.search(r'https?://\S*doi\S*', text, re.IGNORECASE)
        doi_pattern4 = re.search(r'(?<=\bDOI\s)(\S+)', text, re.IGNORECASE)
        if doi_pattern1:
            doi = doi_pattern1.group(1)
        elif doi_pattern2:
            doi = doi_pattern2.group()
        elif doi_pattern3:
            doi = doi_pattern3.group()
        elif doi_pattern4:
            doi = doi_pattern4.group()
        else:
            print("DOI not found in the text.")
            doi = ''
   return {
        'Title': metadata.title if metadata.title else '',
        'DOI': doi,
    }



def extract_pdf_text(pdf_path):
    text_contents = []
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text_contents.append(page.extract_text())
    return text_contents

def sanitize_text(text):
    return text.replace("\n", "").replace("\r", "").replace("\t", "")

def main(pdf_path):
    sticky_notes = extract_sticky_notes(pdf_path)
    text_contents = extract_pdf_text(pdf_path)
    metadata = extract_pdf_metadata(pdf_path)
    file_contents = "".join(text_contents)
    file_contents = sanitize_text(file_contents)
    input_data = {
        "PaperDOI": metadata['DOI'],
        "PaperTitle": metadata['Title'],
        "PaperContents": file_contents,
        "Relations": sticky_notes,
    }
    serialized_input_data = json.dumps(input_data, indent=4)
    metadata['Title'] = sanitize_text(metadata['Title'])
    file_name = f"{metadata['Title']}_{re.sub('[^a-zA-Z0-9]', '_', metadata['DOI'])}.json".replace(" ", "_").replace(":", "")
    file_name = file_name[:50] + ".json"
    with open(f"./auto_generated_inputs_new/{file_name}", "w") as outfile:
        outfile.write(serialized_input_data)


def parse_arguments(args):
    arg_dict = {}
    for arg in args[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            arg_dict[key] = value
    return arg_dict

if __name__ == "__main__":
    cwd = "./coded_papers"
    for file in os.listdir(cwd):
        try:
            main(os.path.join(cwd, file))
        except:
            print("failed on", os.path.join(cwd, file))
