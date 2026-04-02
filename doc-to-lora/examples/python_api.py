# caveat: this interface only supports non-batched inputs
# for batched inference please see `src/ctx_to_lora/modeling/hypernet.py`
import torch

from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

# model loading
checkpoint_path = "trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin"
state_dict = torch.load(checkpoint_path, weights_only=False)
model = ModulatedPretrainedModel.from_state_dict(
    state_dict, train=False, use_sequence_packing=False
)
model.reset()
tokenizer = get_tokenizer(model.base_model.name_or_path)

# prepare data
doc = open("data/sakana_wiki.txt", "r").read()
chat = [{"role": "user", "content": "Tell me about Sakana AI."}]
chat_ids = tokenizer.apply_chat_template(
    chat,
    add_special_tokens=False,
    return_attention_mask=False,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)


# calls after internalization will be influenced by internalized info
model.internalize(doc)

outputs = model.generate(input_ids=chat_ids, max_new_tokens=512)
print(tokenizer.decode(outputs[0]))


# remove internalized info
# model.reset()

# without internalized info, the model will halucinate
# outputs = model.generate(input_ids=chat_ids, max_new_tokens=512)
# print(tokenizer.decode(outputs[0]))
