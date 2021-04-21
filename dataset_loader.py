import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.aspects, self.contexts, self.labels, self.len_aspects, self.len_contexts = self.data
        self.len = len(self.labels)
        
        self.max_aspect_len = self.aspects.shape[1]
        self.max_context_len = self.contexts.shape[1]

        self.aspect_mask = torch.zeros(self.max_aspect_len, self.max_aspect_len)
        self.context_mask = torch.zeros(self.max_context_len, self.max_context_len)

        self.prepare_masks('context')
        self.prepare_masks('aspect')

        self.aspects = torch.from_numpy(self.aspects)
        self.contexts = torch.from_numpy(self.contexts)
        self.labels = torch.from_numpy(self.labels)
        self.len_aspects = torch.from_numpy(self.len_aspects)
        self.len_contexts = torch.from_numpy(self.len_contexts)

        self.aspects = self.aspects.type(torch.LongTensor)
        self.contexts = self.contexts.type(torch.LongTensor)
        self.labels = self.labels.type(torch.LongTensor)
        self.len_aspects = self.len_aspects.type(torch.LongTensor)
        self.len_contexts = self.len_contexts.type(torch.LongTensor)

 
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        aspect = self.aspects[index]
        context = self.contexts[index]
        label = self.labels[index]
        aspect_mask = self.aspect_mask[self.len_aspects[index] - 1]
        context_mask = self.context_mask[self.len_contexts[index] - 1]

        data = (aspect, context, label, aspect_mask, context_mask)
        return data

    def prepare_masks(self, flag):
        if flag == 'context':
            for i in range(self.max_context_len):
                self.context_mask[i, 0:i+1] = 1

        elif flag == 'aspect':
            for i in range(self.max_aspect_len):
                self.aspect_mask[i, 0:i+1] = 1