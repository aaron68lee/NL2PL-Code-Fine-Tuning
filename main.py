import torch
from transformers import (
    RobertaTokenizer, RobertaConfig, RobertaModel, 
    AutoTokenizer, AutoModel,
    RobertaForMaskedLM, pipeline
)

# load pre-trained ROBERTA model from HuggingFace

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")
# tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
# model = AutoModel.from_pretrained("microsoft/codebert-base")
# tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
# model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# obtain NL-PL Embeddings

NL_prompt = "return maximum value"
PL_prompt = "def max(a,b): if a>b: return a else return b"

# tokenize input 

nl_tokens=tokenizer.tokenize(NL_prompt)
code_tokens=tokenizer.tokenize(PL_prompt)

tokens=[tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token] + code_tokens + [tokenizer.eos_token]

print(tokens)
tokens_ids=tokenizer.convert_tokens_to_ids(tokens) #['</s>'])

# obtain token embeddings for semantic code search ...

context_embeddings=model(torch.tensor(tokens_ids)[None,:])[0]
#print(context_embeddings)

# MLM Masked token code prediction
'''
CODE = "if (x is not None) <mask> (x>1)"
fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)

outputs = fill_mask(CODE)
print(outputs)
'''