import pandas as pd
from pprint import pprint
import json


def compare(prediction, ground_truth):
    score_dictionary = dict()
    if prediction["RelationshipClassification"].lower() == ground_truth["RelationshipClassification"].lower():
        score_dictionary["RelationshipClassificationScore"] = 1
        print("RelationshipClassification Score ==> success;", ground_truth["RelationshipClassification"])
    else:
        score_dictionary["RelationshipClassificationScore"] = 0
        print("RelationshipClassification Score ==> actual:", ground_truth["RelationshipClassification"], "; predicted:", prediction["RelationshipClassification"])
    return score_dictionary




# To read the DataFrame and convert back to dictionaries
reloaded_df = pd.read_csv("test_solution.csv")
print(reloaded_df.head())
reloaded_jsons = [json.loads(data) for data in reloaded_df['json_data']]

total_relations = 0
for i in reloaded_jsons:
    # print(i["Relations"][0]["VariableOneName"])
    total_relations += len(i["Relations"])
    # print(i["Relations"][0]["RelationshipClassification"])
print("total relations:", total_relations)

ground_truth_jsons = pd.read_csv("test_solution.csv")
ground_truth_jsons = [json.loads(data) for data in ground_truth_jsons['json_data']]

total_score = 0
total_relations = 0
for i, paper in enumerate(reloaded_jsons):
    ground_truth = ground_truth_jsons[i]
    print(ground_truth["PaperTitle"][:30])
    if len(paper["Relations"]) > len(ground_truth["Relations"]):
        index_max = len(ground_truth["Relations"])
        raise Exception("Prediction and Groundtruth lengths do not match")
    elif len(paper["Relations"]) < len(ground_truth["Relations"]): 
        index_max = len(paper["Relations"])
        raise Exception("Prediction and Groundtruth lengths do not match")
    
    for j, prediction in enumerate(paper["Relations"]):
        total_relations += 1
        relation = ground_truth["Relations"][j]
        total_score += compare(prediction, relation)["RelationshipClassificationScore"]

final_score = total_score / total_relations
print(final_score)