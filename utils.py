import numpy as np
import threading
import queue
from multiprocessing import Process, Queue
from collections import defaultdict
import jax
import jax.numpy as jnp



def make_batch(samples):
    batch = {k:jnp.array(v) for k,v in samples.items()}
    batch['labels'] = batch['input_ids'].copy()
    return batch

class PrefetchDataloaderTread(threading.Thread):
    "Prefetch dataloader for IterableDataset"
    def __init__(self, dataset, batch_size, sequence_length, prefetch_buffer=1, shuffle=True, shuffle_buffer=1000, seed=0):
        super().__init__(daemon=True)
        self.bs = batch_size
        self.seq_len = sequence_length
        self.max_length = batch_size * sequence_length
        self.prefetch_buffer = prefetch_buffer
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.dataset = dataset
        if shuffle:
            shuffled_dataset = dataset.shuffle(shuffle_buffer, seed=self.seed)
            self.seed += 1
            self.ds_iter = iter(shuffled_dataset)
        else:
            self.ds_iter = iter(dataset)
        self.queue = queue.Queue(prefetch_buffer)
        self.rem = defaultdict(list)
        self.start()
        
    def __next__(self):
        batch = self.queue.get()
        return batch
        
    def run(self):
        while True:
            # prepair next batch
            sample = self.rem.copy()
            l = len(sample["input_ids"])
            max_length = self.max_length
            while l < max_length:
                next_sample = next(self.ds_iter)
                l += len(next_sample["input_ids"])
                sample = {k:sample[k]+next_sample[k] for k in next_sample.keys()}
            
            self.rem = {k:v[max_length:] for k,v in sample.items()}
            sample = {k:v[:max_length] for k,v in sample.items()}
            # regroup to shape [bs x seq_len]
            samples = {k:np.array([v[i*self.seq_len:(i+1)*self.seq_len] for i in range(self.bs)]) for k,v in sample.items()}
            
            self.queue.put(make_batch(samples))
    
    def __iter__(self):
        return self


class PrefetchDataloader(Process):
    "Prefetch dataloader for IterableDataset"
    def __init__(self, dataset, batch_size, sequence_length, prefetch_buffer=1, shuffle=True, shuffle_buffer=1000, seed=0):
        super().__init__(daemon=True)
        self.bs = batch_size
        self.seq_len = sequence_length
        self.max_length = batch_size * sequence_length
        self.prefetch_buffer = prefetch_buffer
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.dataset = dataset
        self.make_iter()
        self.queue = Queue(prefetch_buffer)
        self.rem = defaultdict(list)
        self.start()

    def make_iter(self):
        if self.shuffle:
            shuffled_dataset = self.dataset.shuffle(self.shuffle_buffer, seed=self.seed)
            self.seed += 1
            self.ds_iter = iter(shuffled_dataset)
        else:
            self.ds_iter = iter(self.dataset)

    def __next__(self):
        return make_batch(self.queue.get())
        
    def run(self):
        while True:
            # prepair next batch
            sample = self.rem.copy()
            l = len(sample["input_ids"])
            max_length = self.max_length
            while l < max_length:
                try:
                    next_sample = next(self.ds_iter)
                except StopIteration:
                    # reset generator if a pass through dataset is completed
                    self.make_iter()
                l += len(next_sample["input_ids"])
                sample = {k:sample[k]+next_sample[k] for k in next_sample.keys()}
            
            self.rem = {k:v[max_length:] for k,v in sample.items()}
            sample = {k:v[:max_length] for k,v in sample.items()}
            # regroup to shape [bs x seq_len]
            samples = {k:np.array([v[i*self.seq_len:(i+1)*self.seq_len] for i in range(self.bs)]) for k,v in sample.items()}
            
            self.queue.put(samples)
    
    def __iter__(self):
        return self