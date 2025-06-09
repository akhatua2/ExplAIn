from datasets import load_dataset

class GSM8KDataLoader:
    def __init__(self):
        self.dataset = None
        self.train_data = None
        self.test_data = None
        
    def load_data(self):
        """Load the GSM8K dataset"""
        self.dataset = load_dataset("openai/gsm8k", "main")
        self.train_data = self.dataset['train']
        self.test_data = self.dataset['test']
        return self.dataset
    
    def format_prompts_sft_boxed(self, split='train'):

        if self.dataset is None:
            raise ValueError("Please load data first using load_data()")
            
        if split not in ['train', 'test']:
            raise ValueError("Split must be either 'train' or 'test'")
            
        data = self.train_data if split == 'train' else self.test_data

        formatted_data = data.map(lambda x: {
            'prompt': f"""Please solve this math problem step by step. Show your reasoning and put the final answer in\\boxed{{}} format. Only put a single number answer in the box.\n\nProblem: {x['question']}""",
            'completion': ''.join(x['answer'].split('####')[:-1]) +  """\\boxed{""" + x['answer'].split('####')[-1].strip() + """}"""
        })

        if split == 'train':
            self.train_data = formatted_data
        else:
            self.test_data = formatted_data
            
        return formatted_data
        
    def format_prompts_sft(self, split='train'):

        if self.dataset is None:
            raise ValueError("Please load data first using load_data()")
            
        if split not in ['train', 'test']:
            raise ValueError("Split must be either 'train' or 'test'")
            
        data = self.train_data if split == 'train' else self.test_data

        formatted_data = data.map(lambda x: {
            'prompt': f"""Please solve this math problem step by step. Show your reasoning and put the final answer in <answer></answer> tags.\n\nProblem: {x['question']}""",
            'completion': ''.join(x['answer'].split('####')[:-1]) + "<answer>" + x['answer'].split('####')[-1].strip() + "</answer>"
        })

        if split == 'train':
            self.train_data = formatted_data
        else:
            self.test_data = formatted_data
            
        return formatted_data

    def format_prompts(self, split='train'):
        """Format the specified split with instruction prompts
        
        Args:
            split (str): Which split to format ('train' or 'test')
        """
        if self.dataset is None:
            raise ValueError("Please load data first using load_data()")
            
        if split not in ['train', 'test']:
            raise ValueError("Split must be either 'train' or 'test'")
            
        data = self.train_data if split == 'train' else self.test_data
        formatted_data = data.map(lambda x: {
            'prompt': f"Please solve this math problem step by step. Show your reasoning and always put the final number answer in \\boxed{{}} format. Only put a single number answer in the box.\n\nProblem: {x['question']}"
        })
        if split == 'train':
            self.train_data = formatted_data
        else:
            self.test_data = formatted_data
            
        return formatted_data
    

    def format_prompts_llama_few_shot_boxed(self, split='train'):
        """Format the specified split with few-shot instruction prompts
        
        Args:
            split (str): Which split to format ('train' or 'test')
        """
        if self.dataset is None:
            raise ValueError("Please load data first using load_data()")
            
        if split not in ['train', 'test']:
            raise ValueError("Split must be either 'train' or 'test'")
            
        data = self.train_data if split == 'train' else self.test_data
        
        # Get 3 examples for few-shot prompting from training data
        few_shot_examples = self.train_data.select(range(4))  # Take first 3 examples
        
        # Create few-shot examples string
        few_shot_text = ""
        for i, example in enumerate(few_shot_examples):
            # Extract the solution part (before ####) and the final answer
            solution_parts = example['answer'].split('####')
            solution = ''.join(solution_parts[:-1]).strip()
            final_answer = solution_parts[-1].strip()
            
            few_shot_text += f"""
            Problem: {example['question']}

            Answer: {solution}""" +  """\\boxed{""" + final_answer + """}
            
            """

        
        formatted_data = data.map(lambda x: {
            'prompt': f"""Please solve this math problem step by step. Show your reasoning and put the final answer in\\boxed{{}} format. Only put a single number answer in the box

            {few_shot_text}

            Problem: {x['question']}
            Answer:
            """
        })
        
        if split == 'train':
            self.train_data = formatted_data
        else:
            self.test_data = formatted_data
            
        return formatted_data
    
    
    def format_prompts_llama_few_shot(self, split='train'):
        """Format the specified split with few-shot instruction prompts
        
        Args:
            split (str): Which split to format ('train' or 'test')
        """
        if self.dataset is None:
            raise ValueError("Please load data first using load_data()")
            
        if split not in ['train', 'test']:
            raise ValueError("Split must be either 'train' or 'test'")
            
        data = self.train_data if split == 'train' else self.test_data
        
        # Get 3 examples for few-shot prompting from training data
        few_shot_examples = self.train_data.select(range(2))  # Take first 3 examples
        
        # Create few-shot examples string
        few_shot_text = ""
        for i, example in enumerate(few_shot_examples):
            # Extract the solution part (before ####) and the final answer
            solution_parts = example['answer'].split('####')
            solution = ''.join(solution_parts[:-1]).strip()
            final_answer = solution_parts[-1].strip()
            
            few_shot_text += f"""
            Problem: {example['question']}

            Answer: {solution} <answer>{final_answer}</answer>

            """
        
        formatted_data = data.map(lambda x: {
            'prompt': f"""Please solve this math problem step by step. Show your reasoning and put the final answer in <answer></answer> tags.

            {few_shot_text}
            Problem: {x['question']}
            Answer:
            """
        })
        
        if split == 'train':
            self.train_data = formatted_data
        else:
            self.test_data = formatted_data
            
        return formatted_data
    
    def format_prompts_llama(self, split='train'):
        """Format the specified split with instruction prompts
        
        Args:
            split (str): Which split to format ('train' or 'test')
        """
        if self.dataset is None:
            raise ValueError("Please load data first using load_data()")
            
        if split not in ['train', 'test']:
            raise ValueError("Split must be either 'train' or 'test'")
            
        data = self.train_data if split == 'train' else self.test_data
        # formatted_data = data.map(lambda x: {

        #     'prompt': f"""Given the following problem, reason and give a final answer to the problem.

        #     Problem: {x['question']}

        #     Your response should end with "The final answer is [answer]" where [answer] is the response to the problem

        #     Solution:"""
        # })
        formatted_data = data.map(lambda x: {

            'prompt': f"""Please solve this math problem step by step. Show your reasoning and put the final answer in <answer></answer> tags.
            \n\nProblem: {x['question']}""",
        })
        if split == 'train':
            self.train_data = formatted_data
        else:
            self.test_data = formatted_data
            
        return formatted_data
        
    def get_split(self, split='train'):
        """Return the specified split of the dataset
        
        Args:
            split (str): Which split to return ('train' or 'test')
        """
        if self.dataset is None:
            raise ValueError("Please load data first using load_data()")
            
        if split not in ['train', 'test']:
            raise ValueError("Split must be either 'train' or 'test'")
            
        return self.train_data if split == 'train' else self.test_data


# Example usage:
# dataloader = GSM8KDataLoader()
# dataset = dataloader.load_data()
# formatted_train = dataloader.format_prompts(split='train')
# formatted_test = dataloader.format_prompts(split='test')
