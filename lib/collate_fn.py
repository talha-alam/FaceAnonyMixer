import torch
from torch.utils.data._utils.collate import default_collate


def collate_fn(batch):
    """Custom collate that handles mixed tensor/string batches.

    DataLoader items returned by CelebAHQ and VGGFace2 contain image tensors
    alongside file-path strings.  PyTorch's default collate cannot stack
    strings into a tensor, so we handle them separately here.

    Args:
        batch (list): list of items, each being a list whose elements are
                      either ``torch.Tensor`` or ``str``.

    Returns:
        list: collated batch where tensors are stacked and strings remain
              as a plain Python list.
    """
    collated = []
    for i, elem in enumerate(zip(*batch)):
        if isinstance(elem[0], torch.Tensor):
            collated.append(torch.stack(list(elem), dim=0))
        elif isinstance(elem[0], str):
            collated.append(list(elem))
        else:
            collated.append(default_collate(list(elem)))
    return collated
