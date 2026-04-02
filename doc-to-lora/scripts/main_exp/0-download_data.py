from huggingface_hub import snapshot_download

if __name__ == "__main__":
    self_gen_data_dir = "./data/raw_datasets/self_gen/"
    snapshot_download(
        "SakanaAI/self_gen_qa_d2l",
        repo_type="dataset",
        local_dir=self_gen_data_dir,
        # we can filter based on model by using the `allow_patterns` argument
        # based on https://huggingface.co/datasets/SakanaAI/self_gen_qa_d2l/tree/main
        # we can use
        # - `Qwen` for downloading the data for `Qwen/Qwen3-4B-Instruct-2507`
        # - `google`  for downloading the data for `google/gemma-2-2b-it`
        # - `mistralai` for downloading the data for `mistralai/Mistral-7B-Instruct-v0.2`
        #
        # allow_patterns="google/*", # downloading the data for `google/gemma-2-2b-it`
    )
