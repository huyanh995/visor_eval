import os
import numpy as np
from PIL import Image
import sys
from glob import glob


class Results(object):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.uneven_length = []

    def _read_mask(self, sequence, frame_id):
        try:
            mask_path = os.path.join(self.root_dir, sequence, f'{frame_id}.png')
            return np.array(Image.open(mask_path))
        except IOError as err:
            sys.stdout.write(sequence + " frame %s not found!\n" % frame_id)
            sys.stdout.write("The frames have to be indexed PNG files placed inside the corespondent sequence "
                             "folder.\nThe indexes have to match with the initial frame.\n")
            sys.stderr.write("IOError: " + err.strerror + "\n")
            sys.exit()

    def read_masks(self, sequence, masks_id):
        res_masks_id = [os.path.basename(x).replace('.png', '') for x in sorted(glob(os.path.join(self.root_dir, sequence, '*.png')))]
        if len(res_masks_id) != len(masks_id):
            self.uneven_length.append(sequence)
            # print(f'Warning: number of masks in {sequence} is different from the number of masks in the ground truth. {len(res_masks_id)} != {len(masks_id)}')

        mask_0 = self._read_mask(sequence, res_masks_id[0]) # Read first mask for know shape
        masks = np.zeros((len(res_masks_id), *mask_0.shape))
        for ii, m in enumerate(res_masks_id):
            masks[ii, ...] = self._read_mask(sequence, m)

        num_objects = int(np.max(masks))
        tmp = np.ones((num_objects, *masks.shape))
        tmp = tmp * np.arange(1, num_objects + 1)[:, None, None, None]
        masks = (tmp == masks[None, ...]) > 0
        return masks
