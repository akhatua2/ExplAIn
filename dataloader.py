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
            'prompt': f"Please solve this math problem step by step. Show your reasoning and put the final answer in <answer></answer> tags.\n\nProblem: {x['question']}"
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
