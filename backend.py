from transformersimport AutoTokenizer
import torch
from model import Seq2SeqEncDec

src_sent_tokenizer = AutoTokenizer.from_pretrained("google-T5/T5-base")

