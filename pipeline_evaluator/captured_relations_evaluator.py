"""
General purpose evaluator should take in a YAML file which specifies the parameters: type of model, evaluation dataset, type of scoring function, and supporting tools combination
Run the appropriate pipeline with these parameters.
Calculate the score.
Save results, score, predictions(inputs and outputs), other relevant information, in evaluation outputs.
Run with: python -m pipeline_evaluator.captured_relations_evalu
ator
"""
import pathlib
from pipelines.captured_relations_pipeline.captured_relations_pipeline import captured_relations_pipeline
import json
import os
from copy import deepcopy

def compare(prediction, ground_truth):
    score_dictionary = dict()
    if prediction["RelationshipClassification"].lower() == ground_truth["RelationshipClassification"].lower():
        score_dictionary["RelationshipClassificationScore"] = 1
        print("RelationshipClassification Score ==> success;", ground_truth["RelationshipClassification"])
    else:
        score_dictionary["RelationshipClassificationScore"] = 0
        print("RelationshipClassification Score ==> actual:", ground_truth["RelationshipClassification"], "; predicted:", prediction["RelationshipClassification"])
    return score_dictionary

def evaluate_one_paper(input_file_path, settings_path, strict_length=True, verbose=False, debug_path=None):
    # Read evaluation dataset
    with open(input_file_path) as f:
        ground_truth = json.load(f)
        print("datatype of ground truth", type(ground_truth))    

    # extract_relationships based on settings (which is text and nothing else)
    predictions = captured_relations_pipeline(input_file_path, settings_path=settings_path, debug_path=debug_path)

    # strip full text from ground truth to prevent accidental printing of full text
    ground_truth["PaperContents"] = ""

    # compare to obtain score
    if len(predictions["Relations"]) > len(ground_truth["Relations"]):
        index_max = len(ground_truth["Relations"])
        if strict_length:
            raise Exception("Prediction and Groundtruth lengths do not match")
    elif len(predictions["Relations"]) < len(ground_truth["Relations"]): 
        index_max = len(predictions["Relations"])
        if strict_length:
            raise Exception("Prediction and Groundtruth lengths do not match")
    else:
        index_max = len(predictions["Relations"])
        print("Number of relations in ground truth and predictions match.")

    results: list[dict] = list()
    for relation_index in range(index_max):
        relation = ground_truth["Relations"][relation_index]
        prediction = predictions["Relations"][relation_index]
        results.append(compare(prediction, relation))

    aggregate_results = dict()
    aggregate_results["RelationshipClassificationScore"] = 0
    for x, result in enumerate(results):
        aggregate_results["RelationshipClassificationScore"] += result["RelationshipClassificationScore"]
    aggregate_results["RelationshipClassificationScore"] /= len(results)
    print(aggregate_results)

    with open(EVAL_OUTPUTS_PATH, "a+") as f:
        f.write(f"\nResults for {input_file_path.name}\n")
        f.write(json.dumps(aggregate_results, indent=2))
        f.write("\n")
        f.write("Predictions\n")
        f.write(json.dumps(predictions, indent=2))
        f.write("\n")
        f.write("Ground Truth\n")
        f.write(json.dumps(ground_truth, indent=2))
        f.write("\n")
    aggregate_results["file"] = input_file_path.name
    return aggregate_results

if __name__ == "__main__":
    # Paths for evaluator module
    DATASET_PATH = pathlib.Path("pipeline_evaluator/full_dataset/")
    INPUT_FILE_PATH = pathlib.Path("pipeline_evaluator/standard_dataset/test_paper_2.json")
    # Paths for pipeline module
    SETTINGS_PATH = pathlib.Path("./pipelines/captured_relations_pipeline/pipeline_settings.yaml")
    DEBUG_PATH = pathlib.Path("./pipelines/captured_relations_pipeline/debug_outputs")
    EVAL_OUTPUTS_PATH = pathlib.Path("./pipelines/captured_relations_pipeline/eval_results/results.txt")
    # If True, use DATASET_PATH and evaluate on DATASET_PATH directory. Else use INPUT_FILE_PATH and evaluate on INPUT_FILE_PATH file.
    MULTIPAPER = True

    if MULTIPAPER:
        with open(EVAL_OUTPUTS_PATH, "w") as f:
            f.write(f"New multi file evaluation source from path: {DATASET_PATH}")
        dir, _, files = next(os.walk(DATASET_PATH))
        full_evaluator_aggregate_results = []
        for file in files: 
            print("\n\nEvaluating: ", file)
            result = evaluate_one_paper(pathlib.Path(dir)/pathlib.Path(file), settings_path=SETTINGS_PATH, verbose=True, debug_path=DEBUG_PATH)
            full_evaluator_aggregate_results.append(result)
        with open(EVAL_OUTPUTS_PATH, "a+") as f:
            f.write("\n\nAggregated_Results:\n")
            for i in full_evaluator_aggregate_results:
                f.write(f"{i['file']}\n")
                f.write(f"{i}\n")
    else:
        with open(EVAL_OUTPUTS_PATH, "w") as f:
            f.write(f"New single file evaluation")
        result = evaluate_one_paper(INPUT_FILE_PATH, settings_path=SETTINGS_PATH, verbose=True, debug_path=DEBUG_PATH)