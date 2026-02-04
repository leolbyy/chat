import os
import re
import json
import random
from utils.common import get_base_dir
from nanochat.sample_eval import get_response
from datasets import load_dataset

BASE_DIR = get_base_dir()

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"
def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS

# User message templates for data augmentation
USER_MSG_TEMPLATES = [
    "How many {letter} are in the word {word}",
    "How many {letter} are in {word}",
    "Count the number of {letter} in {word}",
    "How many times does {letter} appear in {word}",
    "What's the count of {letter} in {word}",
    "In the word {word}, how many {letter} are there",
    "How many letter {letter} are in the word {word}",
    "Count how many {letter} appear in {word}",
    "Tell me the number of {letter} in {word}",
    "How many occurrences of {letter} are in {word}",
    "Find the count of {letter} in {word}",
    "Can you count the {letter} letters in {word}",
    "What is the frequency of {letter} in {word}",
    "How many {letter}s are in {word}",
    "How many {letter}'s are in {word}",
    "Count all the {letter} in {word}",
    "How many times is {letter} in {word}",
    "Number of {letter} in {word}",
    "Total count of {letter} in {word}",
    "How many {letter} does {word} have",
    "How many {letter} does {word} contain",
    "What's the number of {letter} in {word}",
    "{word} has how many {letter}",
    "In {word}, count the {letter}",
    "How many {letter} appear in {word}",
    "Count the {letter} in {word}",
    "Give me the count of {letter} in {word}",
    "How many instances of {letter} in {word}",
    "Show me how many {letter} are in {word}",
    "Calculate the number of {letter} in {word}"
]

class Task:
    def __init__(self, split):
        assert split in ('train', 'test'), f'split must be in train | test. got {split}'

    def get_example(self):
        raise NotImplementedError
    
    def get_num_examples(self, size=-1):
        if size > 0:
            return min(len(self.ds), size)
        return len(self.ds)

class GSM8K(Task):
    def __init__(self, subset, split):
        super().__init__(split)
        assert subset in ('main', 'socratic')
        self.ds = load_dataset('openai/gsm8k', subset, split=split).shuffle(seed=2026)
    
    def get_example(self, idx):
        example = self.ds[idx]
        question = example['question']
        answer = example['answer']
        # remove calculator indicator
        question = ''.join(re.split(r'<<|>>', question))
        question += '\n'
        answer = ''.join(re.split(r'<<|>>', answer))
        # answer, _ = ANS_RE.search(answer)

        # render into conversation
        messages = [{'content': question, 'role': 'user'}, {'content': answer, 'role': 'assistant'}]
        return messages

class MMLU(Task):
    def __init__(self, subset, split):
        super().__init__(split)
        assert subset in ('all', 'auxiliary_train')
        self.ds = load_dataset('cais/mmlu', subset, split=split).shuffle(seed=2026)
        if subset == 'auxiliary_train':
            self.ds = self.ds.map(lambda x: x['train'])
    
    def get_example(self, idx):
        example = self.ds[idx]
        question = example['question']
        choices = example['choices']
        answer = int(example['answer'])

        messages = []

        letters = ('A', 'B', 'C', 'D')
        prompt = ''
        prompt = prompt + f'Question: {question}\n'
        prompt = prompt + f'Choices: '
        choices = ''.join([f'{letters[i]}:{choices[i]}' for i in range(4)])
        prompt = prompt + choices + '\n'
        # prompt = prompt + 'Respond only with letter of the correct answer.\n'

        messages.append({'content': prompt, 'role': 'user'})
        messages.append({'content': f'The correct answer is {letters[answer]} #### {letters[answer]}', 'role': 'assistant'})
        
        return messages

class SmolTalk(Task):
    def __init__(self, split, size=-1):
        super().__init__(split)
        self.ds = load_dataset('HuggingFaceTB/smol-smoltalk', split=split).shuffle(seed=2026)
    def get_example(self, idx):
        messages = self.ds[idx]['messages']
        return messages
        
class SimpleSpelling(Task):
    def __init__(self, split, size=-1):
        super().__init__(split)
        
        filepath = os.path.join(BASE_DIR, 'words_alpha.txt')
        with open(filepath, 'r') as f:
            self.ds = [line.strip() for line in f.readlines()]
        rng = random.Random(2026)
        rng.shuffle(self.ds)

        ratio = 0.9 # 90% for train, 10% for test
        split_idx = int(len(self.ds) * ratio)

        if split == 'train':
            self.ds = self.ds[:split_idx]
        else:
            self.ds = self.ds[split_idx:]
        
    def get_example(self, idx):
        word = self.ds[idx]
        ans = ','.join(word)
        messages = []
        prompt = f'Please spell word {word}\n'
        messages.append({'content': prompt, 'role': 'user'})
        messages.append({'content': ans, 'role': 'assistant'})
    
        return messages

class SpellingBee(Task):
    def __init__(self, split, size=-1):
        super().__init__(split)
        
        filepath = os.path.join(BASE_DIR, 'words_alpha.txt')
        with open(filepath, 'r') as f:
            self.ds = [line.strip() for line in f.readlines()]
        rng = random.Random(2026)
        rng.shuffle(self.ds)

        ratio = 0.9 # 90% for train, 10% for test
        split_idx = int(len(self.ds) * ratio)

        if split == 'train':
            self.ds = self.ds[:split_idx]
        else:
            self.ds = self.ds[split_idx:]
    def get_example(self, idx):
        word = self.ds[idx]
        rng = random.Random(2026 + idx)
        letter = rng.choice(word)
        answer = word.count(letter)

        messages = []
        template = rng.choice(USER_MSG_TEMPLATES)
        prompt = template.format(word=word, letter=letter) + '\n'
        messages.append({'content': prompt, 'role': 'user'})
        
        answer_string = f'There are in total {answer} {letter} in word {word}. #### {answer}'
        messages.append({'content': answer_string, 'role': 'assistant'})
        return messages

class CustomJSON(Task):
    def __init__(self, filepath, split):
        super().__init__(split)
        # additional assert, no test split in identity file
        assert split == 'train'
        with open(filepath, 'r') as f:
            self.ds = [line.strip() for line in f.readlines()]
        rng = random.Random(2026)
        rng.shuffle(self.ds)
    
    def get_example(self, idx):
        messages = self.ds[idx]
        messages = json.loads(messages)
        return messages

class ARC:
  def __init__(self, subset, split):
    super().__init__(split)
    assert subset in ('ARC-Easy', 'ARC-Challenge')
    self.ds = load_dataset('allenai/ai2_arc', subset, split=split).shuffle(seed=2026)
  
  def get_example(self, idx):
    example = self.ds[idx]
    question = example['question']
    choices = example['choices']['text']
    letters = example['choices']['label']
    answer = example['answerKey']

    messages = []

    prompt = ''
    prompt = prompt + f'Question: {question}\n'
    prompt = prompt + f'Choices: '
    choices = ''.join([f'{letters[i]}:{choices[i]}' for i in range(len(letters))])
    prompt = prompt + choices + '\n'
    # prompt = prompt + 'Respond only with letter of the correct answer.\n'

    messages.append({'content': prompt, 'role': 'user'})
    messages.append({'content': f'The correct answer is {answer} #### {answer}', 'role': 'assistant'})

    return messages



class TaskMixture:
    def __init__(self, tasks):
        self.tasks = tasks
    
    def __getitem__(self, task_idx):
        return self.tasks[task_idx]

    def get_example(self, task_idx, idx):
        return self.tasks[task_idx].get_example(idx)
    
    def get_num_examples(self):
        return [task.get_num_examples() for task in self.tasks]
    
    def get_num_examples_total(self):
        return sum(self.get_num_examples())




def eval_task(task_name, model, tokenizer, max_problems=-1, k_shot=3):
    assert task_name in ('mmlu', 'arc-easy'), f'currently eval task only support multiple choice tasks. expect mmlu | arc-easy, got {task_name}'
    if task_name == 'mmlu':
        task = MMLU(subset='all', split='test')
    elif task_name == 'arc-easy':
        task = ARC(subset='ARC-Easy', split='test')
    
    if max_problems == -1:
        max_problems = task.get_num_examples()

    
    correct_count = 0
    
    for idx in range(max_problems):
        example = task.get_example(idx)
        input_token_ids, _ = tokenizer.render_conversation(example[:-1]) # remove answer
        answser = extract_answer(example[-1]['content']) # get answer from last assistant message

        assistant_end = tokenizer.encode_special('<|assistant_end|>')
        bos = tokenizer.encode_special('<|bos|>')
        max_tokens = 128
        for _ in range(k_shot):
            output = []
            for next_token in model.generate(input_token_ids.to(model.device), max_tokens=max_tokens, temperature=1.0, top_k=10):
                if (next_token == assistant_end or next_token == bos) or len(output) >= max_tokens:
                    break
                else:
                    output.append(next_token)
            output_str = tokenizer.decode(output)
            output_answer = extract_answer(output_str)
            if output_answer == INVALID_ANS:
                continue
            else:
                if output_answer == answser:
                    correct_count += 1
                break
    accuracy = correct_count / max_problems
    return accuracy








