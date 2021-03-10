
from transformers import RobertaTokenizer, RobertaConfig

my_tok_path = "model/tok"
# load tokenizer and model from artifacts in model context
tokenizer = RobertaTokenizer.from_pretrained('roberta-base', cache_dir=my_tok_path)
config = RobertaConfig.from_pretrained('roberta-base', cache_dir=my_tok_path)
tokenizer.save_pretrained(my_tok_path)
config.save_pretrained(my_tok_path)

txt="hello world"

mytok=RobertaTokenizer.from_pretrained(my_tok_path, local_files_only=True)

assert tokenizer.encode(txt, add_special_tokens=True) == mytok.encode(txt, add_special_tokens=True)

