import logging
from typing import List
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {}
    # TODO: your code here
    # raise NotImplementedError
    for key in ['text_encoded', 'spectrogram', 'audio']:
        max_len = max(list(map(lambda x: x[key].shape[-1], dataset_items)))
        result_batch[key] = torch.cat(list(
            map(lambda x: F.pad(x[key], (0, max_len - x[key].shape[-1])), dataset_items)
        ), dim=0)

    for key in ['text', 'duration', 'audio_path']:
        result_batch[key] = [x[key] for x in dataset_items]
    result_batch['spectrogram'] = result_batch['spectrogram'].permute(0, 2, 1)
    result_batch['text_encoded_length'] = torch.tensor(list(map(lambda x: x['text_encoded'].shape[-1], dataset_items)), dtype=torch.int32)
    result_batch['spectrogram_length'] = torch.tensor(list(map(lambda x: x['spectrogram'].shape[-1], dataset_items)), dtype=torch.int32)
    return result_batch