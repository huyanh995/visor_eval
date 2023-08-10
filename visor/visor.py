import os
from glob import glob
from collections import defaultdict
import numpy as np
from PIL import Image

class VISOR:

    def __init__(self, root):
        self.img_root = os.path.join(root, 'JPEGImages')
        self.mask_root = os.path.join(root, 'GT')

        # Load sequences from dataset
        self.sequences = defaultdict(dict)
        sequences_names = [x for x in os.listdir(self.img_root) if not x.startswith('.')] # remove hidden files
        for seq in sequences_names:
            images = sorted(glob(os.path.join(self.img_root, seq, '*.jpg')))
            masks = sorted(glob(os.path.join(self.mask_root, seq, '*.png')))
            if len(images) == 0 or len(masks) == 0:
                raise FileNotFoundError(f'Images or masks for sequence {seq} not found.')

            self.sequences[seq]['images'] = images
            self.sequences[seq]['masks'] = masks

    # def get_frames(self, sequence):
    #     for image, mask in zip(self.sequences[sequence]['images'], self.sequences[sequence]['masks']):
    #         image = np.array(Image.open(image)) # Why you need image?
    #         mask = np.array(Image.open(mask))
    #         yield image, mask

    def get_all_masks(self, sequence):
        # Get all masks from a sequence
        _mask = np.array(Image.open(self.sequences[sequence]['masks'][0]))
        masks = np.zeros((len(self.sequences[sequence]['masks']), *_mask.shape))
        masks_id = []
        for i, obj in enumerate(self.sequences[sequence]['masks']):
            masks[i, ...] = np.array(Image.open(obj))
            masks_id.append(''.join(obj.split('/')[-1].split('.')[:-1]))

        masks_void = np.zeros_like(masks)

        # Separate void and object masks:
        for i in range(masks.shape[0]):
            masks_void[i, ...] = masks[i, ...] == 255   # bool array (seq_len, H, W)
            masks[i, masks[i, ...] == 255] = 0          # change 255 to 0 (seq_len, H, W)

        # Check the labels are consecutive
        labels = np.unique(masks)
        if not np.all(np.diff(labels) == 1):
            print(f'Labels are not consecutive {sequence}')

        num_objects = int(np.max(labels))
        tmp = np.ones((num_objects, *masks.shape)) # (num_objects, seq_length, H, W)
        tmp = tmp * np.arange(1, num_objects + 1)[:, None, None, None] # One hot encoder for each object
        masks = (tmp == masks[None, ...]) # Note: no mask for background at index 0
        masks = masks > 0

        return masks, masks_void, masks_id

    def get_sequences(self):
        for seq in self.sequences:
            yield seq

    def __len__(self):
        return len(self.sequences)

