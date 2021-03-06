from keras.utils import Sequence
from keras.preprocessing.image import load_img, img_to_array

import Augmentor
import numpy as np

class Generator(Sequence):

    def __init__(self, image_filenames, labels, batch_size, pipeline):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size
        self.p = pipeline

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        if self.p:
            to_ops = []

            for path in batch_x:
                to_ops.append(load_img(path))

            x_out = []

            for img in to_ops:
                this = [img]
                for operation in self.p.operations:
                    r = np.round(np.random.uniform(0, 1, 1), 1)[0]
                    if r <= operation.probability:
                        this = operation.perform_operation(this)
                to_out = img_to_array(this[0])
                to_out /= 255.
                # patch attempt
                x_len, y_len = 76, 76
                x_start = np.random.randint(0, 256-x_len)
                y_start = np.random.randint(0, 256-y_len)
                to_out = to_out[x_start:x_start+x_len, y_start:y_start+y_len, :]


                x_out.append(to_out)
                img.close()
            return np.array(x_out), np.array(batch_y)
        else:
            x_out = []

            for path in batch_x:
                img = load_img(path)
                arr = img_to_array(img)
                arr /= 255.
                # patch attempt
                x_len, y_len = 76, 76
                x_start = np.random.randint(0, 256-x_len)
                y_start = np.random.randint(0, 256-y_len)
                arr = arr[x_start:x_start+x_len, y_start:y_start+y_len, :]

                x_out.append(arr)
            return np.array(x_out), np.array(batch_y)