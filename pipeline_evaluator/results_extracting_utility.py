import json
import os


def build_search_strings(base_search_string: str):
    search_strings = ["\n" + base_search_string + "\n"]
    search_strings += [base_search_string + "\n\n"]
    # + :
    search_strings += ["\n" + base_search_string + ":" + "\n"]
    search_strings += [base_search_string + ":" + "\n\n"]
    search_strings += [base_search_string + ":" + "\n"]
    search_strings += [base_search_string + ":"]
    return search_strings


def main(paper_testfile_path):
    print("\033[95mProcessing paper")
    file = open(paper_testfile_path)
    data = json.load(file)
    content_lowercase = data["content"].lower()

    index = -1
    index_search_strings = (
        build_search_strings("results")
        + build_search_strings("conclusions")
        + build_search_strings("conclusion")
    )
    for search_string in index_search_strings:
        index = content_lowercase.rfind(search_string)
        if index > len(content_lowercase) * 2 / 5:
            break
        elif index > -1:
            print(
                f"\033[93mSearch string found too early in content: {repr(search_string)} {index} / {len(content_lowercase)} \033[0;0m"
            )

    if index == -1:
        print(f"\033[91m{data["title"]}\033[0;0m: first index not found")
        print("\n")
        return

    last_index = len(content_lowercase)
    last_index_search_strings = (
        build_search_strings("discussion")
        + build_search_strings("references")
        + build_search_strings("acknowledgements")
        + build_search_strings("acknowledgement")
    )
    for search_string in last_index_search_strings:
        temp = content_lowercase[index:].rfind(search_string)
        if temp > -1 and temp < last_index:
            last_index = temp

    if last_index < len(content_lowercase):
        last_index += index

    result = content_lowercase[index:last_index]
    if len(result) < 1:
        print(f"\033[91m{data["title"]}\033[0;0m: result empty")
    elif len(result) < 150:
        print(f"\033[91m{data["title"]}\033[0;0m: result too short \033[93m({len(result)})\033[0;0m")
    else:
        print(f"\033[32m{data["title"]}\033[0;0m")
        print(f"Index: {index}, Last Index: {index}")
        print(f"Result Length: {len(result)}")
    print("\n")
    return


if __name__ == "__main__":
    cwd = "./pipeline_evaluator/full_dataset"
    for file_name in os.listdir(cwd):
        try:
            main(os.path.join(cwd, file_name))
        except:
            print("failed on", os.path.join(cwd, file_name))

        # break
