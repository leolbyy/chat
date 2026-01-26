from bpe.tokenizer import BaseTokenizer
from utils.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit
tokenizer = BaseTokenizer.load_from_directory('/home/boyu/learn/chat/tokenizer')
x = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, 2, 4, split='val')

for i in range(1000000):
    a, b, _= next(x)
    if (b < 0).any():
        print(a, b)
        break
print(a, b)