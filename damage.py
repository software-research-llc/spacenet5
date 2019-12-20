import numpy as np
import logging
import infer
import train
import deeplabmodel
import settings as S
import tensorflow as tf
import plac
import unet
import sys
import tensorflow as tf
import os
import sys
import skimage
import score
import random
from flow import Dataflow, BuildingDataflow, get_training_files, get_validation_files

logger = logging.getLogger(__name__)


def mode(ary):
    ary = ary.ravel()
    counts = {}
    for val in ary:
        if counts.get(val, None) is None:
            counts[val] = 1
        else:
            counts[val] += 1

    # this can happen if we were passed an image with no values, and is
    # technically an error condition; returning "no damage" is the best
    # we can do without causing the entire training process to die
    if len(ary) == 0:
        return 1

    return sorted(counts, key=lambda x: counts[x])[-1]


class ModelShell(unet.MotokimuraUnet):
    def __init__(self):
        self.model = None

def build_model(backbone=S.ARCHITECTURE,
                train=False,
                classes=5):

    model = ModelShell()
    model.model = tf.keras.applications.ResNet50(classes=classes, include_top=True, input_shape=S.DMG_SAMPLESHAPE, weights=None)
    return model


def load_weights(model, save_file=S.DMG_MODELSTRING):
    model.model.load_weights(save_file)
    logger.info("Loaded {} successfully.".format(save_file))
    return model


def extract_patches(pre, post, mask, return_masks=False, return_dict=False, max_x=S.DAMAGE_MAX_X, max_y=S.DAMAGE_MAX_Y):
    """
    Extract all portions of `pre` and `post` corresponding to contiguous areas in `mask`.  The ground truth is
    determined by taking the highest frequency damage class found among the values in contiguous regions of 'mask'
    (since it's possible for multiple buildings to have no gap between them, the mode value among buildings is the
    one returned).

    Returns:
      preboxes: variably sized portions of the pre-disaster image (matching areas in input mask)
      postboxes: variably sized portions of the post-disaster image (matching areas in input mask)
      klasses: one-hot encoded list of damage classes, indices of which correspond to indices in preboxes/postboxes
      masks (optional): variably sized portions of the mask corresponding to preboxes/postboxes (e.g. for debugging)

        ** or if return_dict=True **

      retdict (optional): a dictionary with all of the above, plus the (x,y) coords of objs within the mask
    """
    preboxes = []
    postboxes = []
    klasses = []
    masks = []
    xys = []
    # extract individual buildings from the mask
    rectangles = infer.bounding_rectangles(mask)
    for rect in rectangles:
        # store a view for each building rectangle (region of interest) that was found
        x,y = rect
        # skip objects that are smaller than 5 pixels
        if x.stop-x.start <= 5 or y.stop-y.start <= 5:
            continue
        # also skip any objects that are larger than the net's input size
        if x.stop-x.start >= max_x:
            x = slice(x.start, x.start + max_x - 1)
        if y.stop-y.start >= max_y:
            y = slice(y.start, y.start + max_y - 1)

        prebox = pre[x.start:x.stop,y.start:y.stop]
        postbox = post[x.start:x.stop,y.start:y.stop]
        retmask = mask[x.start:x.stop,y.start:y.stop]
        #import pdb; pdb.set_trace()
        klass = mask[x.start:x.stop,y.start:y.stop]
        klass = mode(klass[np.nonzero(klass)])
        if klass > 5 or klass < 0:
            logger.warning("Unrecognized class! Setting to appropriate value")
            klass = 1
        preboxes.append(np.ascontiguousarray(prebox))#.astype(np.uint8))
        postboxes.append(np.ascontiguousarray(postbox))#.astype(np.uint8))
        masks.append(retmask)
        klass_one_hot = [0] * 5
        klass_one_hot[klass-1] = 1
        klasses.append(klass_one_hot)
        xys.append((x,y))

#    import pdb; pdb.set_trace()
    preboxes = np.asarray(preboxes)
    postboxes = np.asarray(postboxes)
    # return all of the above plus the (x,y) bounding boxes
    if return_dict:
        retdict = { 'bbox': xys, 'class': klasses, 'mask': np.array(masks, copy=False),
                    'prebox': preboxes, 'postbox': postboxes }
        return retdict

    # return all the (pre-disaster, post-disaster) ROIs, their corresponding damage classes,
    # and optionally the mask we used.
    # If extraction algorithm failed, return empty values
    if return_masks is True:
        if len(preboxes) < 1:
            return (np.empty(0),np.empty(0)),[],np.empty(0)
        return (np.array(preboxes, copy=False), np.array(postboxes, copy=False)), klasses, np.array(masks, copy=False)
    else:
        if len(preboxes) < 1:
            return (np.empty(0),np.empty(0)),[]
        return (np.array(preboxes, copy=False), np.array(postboxes, copy=False)), klasses


def get_buildings(preboxes, postboxes):
        buildings = []
        for i in range(len(preboxes)):
            prebox = preboxes[i]
            postbox = postboxes[i]
            dim = prebox.shape

            concat = np.zeros(S.INPUTSHAPE, dtype=prebox.dtype)#dim[1] * 2, dim[2] * 2, dim[3])
            concat[0:dim[0],0:dim[1],:] = prebox
            concat[dim[0]:dim[0]*2,0:dim[1],:] = postbox

            buildings.append(concat)

        return buildings

class DamageDataflow(Dataflow):
    """
    Exactly like a Dataflow() object (see flow.py), but with its own __getitem__() method
    suited to damage classification.
    """
    def __init__(self, return_masks=True, *args, **kwargs):
        super(DamageDataflow, self).__init__(buildings_only=True, *args, **kwargs)
        self.return_masks = return_masks
        self.buildings_only = True

    def __getitem__(self, idx):
        """
        Get an item by index.

        Returns:
          preboxes: variably sized portions of the pre-disaster image corresponding to a single building (or contiguous ones).
          postboxes: variably sized portions of the post-disaster image corresponding to a single building (or contiguous ones).
          klasses: one-hot encoded list of damage classes, indices of which correspond to indices in preboxes/postboxes.
          masks: variably sized portions of the mask; these are what dictated the regions of each prebox/postbox
          buildings: constant-sized square of the prebox and postbox content stacked vertically with the rest zero padded.
        """
        (x,y), mask = Dataflow.__getitem__(self, idx, preprocess=False, return_postmask=True)
        mask = infer.weave_pred(mask)
        x = infer.weave(x)
        y = infer.weave(y)

        (preboxes, postboxes), klasses, masks = extract_patches(x, y,
                                                                mask,
                                                                return_masks=self.return_masks)

        if len(preboxes) == 0:
            return self[idx+1]
        # Make the pre- and post- disaster building locations one INPUTSHAPE-sized image by
        # placing them next to each other in the top left corner and padding the remainder
        # with zeros
        buildings = []
        for i in range(len(preboxes)):
            prebox = preboxes[i]
            postbox = postboxes[i]
            dim = prebox.shape

            concat = np.zeros(S.INPUTSHAPE, dtype=prebox.dtype)#dim[1] * 2, dim[2] * 2, dim[3])
            concat[0:dim[0],0:dim[1],:] = prebox
            concat[dim[0]:dim[0]*2,0:dim[1],:] = postbox

            buildings.append(concat)

        # return the (pre-disaster, post-disaster) ROIs, corresponding damage classes (ground truth),
        # the mask, and pre + post image buildings laid out in one image next to each other (one image
        # per set).
        return (preboxes, postboxes), klasses, masks, buildings


class BuildingDataflow_old(tf.keras.utils.Sequence):
    """
    A dataflow to train on building damage classification.  Expects buildings to exist 
    in the `topdir` passed to __init__(), and to be .png format files of pre- and post-
    disaster images laid out on top of each other.  Each pre- and post-disaster image
    pair has multiple building images that use the pre-disaster name only for the directory
    in which to find the buildings.  The target class for a given image is in the filename,
    e.g. `topdir/hurricane-harvey-pre-00001/0:1.png` is the first building in hurricane
    harvey image 00001, and the buildings in the file belong to the 'minor-damage' class.
    """
    def __init__(self, topdir="buildings", batch_size=50, train=False, validate=False, shuffle=True):
        self.files = []
        self.topdir = topdir
        self.batch_size = batch_size

        dirs = os.listdir(topdir)
        for dir in dirs:
            files = os.listdir(os.path.join(topdir, dir))
            for file in files:
                self.files.append(os.path.abspath(os.path.join(topdir, dir, file)))
        
        self.files.sort()

        if train:
            self.files = self.files[:int(np.floor(len(self.files)*0.9))]
        elif validate:
            self.files = self.files[int(np.floor(len(self.files)*0.9)):]
        else:
            raise Exception("Should be either train or validate")

        if shuffle:
            random.shuffle(self.files)

    def __len__(self):
        length = int(np.ceil(len(self.files) / float(self.batch_size)))
        return length

    def __getitem__(self, idx):
        x = []
        y = []
        for filename in self.files[idx*self.batch_size:(idx+1)*self.batch_size]:
            try:
                img = skimage.io.imread(filename)
            except Exception as exc:
                logger.error(str(exc))
                continue

            klass = int(filename.split(":")[-1][0])
            onehot = [0] * 5
            onehot[klass] = 1

            x.append(img)
            y.append(onehot)

        return np.array(x, copy=False), np.array(y, copy=False)


#FIXME: memory leak somewhere
def epoch(model, train_seq, val_seq, noaction=False, step=16):
    MEMPROFILE = False
    import gc
    for j in range(len(train_seq)):
        try:
            (_, _), klasses, _, buildings = train_seq[j]
        except Exception as exc:
            logger.error(str(exc))
            continue

    #for (pre,post), klasses, mask, buildings in train_seq:
        buildings = np.array(buildings)
        klasses = np.array(klasses)
        for i in range(0, len(buildings), step):
            if i+step > len(buildings):
                if noaction:
                    buf1, buf2 = buildings[i:], klasses[i:]
                    logger.info(f"{j}:{i}: {len(buf1)} samples accessed successfully.")
                    continue
                model.fit(buildings[i:], klasses[i:],
                                    verbose=2, shuffle=False, use_multiprocessing=False)
            if noaction:
                buf1, buf2 = buildings[i:i+step], klasses[i:i+step]
                logger.info(f"{j}:{i}: {len(buf1)} samples accessed successfully.")
                continue
            model.fit(buildings[i:i+step], klasses[i:i+step],
                                verbose=2, shuffle=False, use_multiprocessing=False)

        if j % 100 == 0:
            num_uncollectable = gc.collect(2)
            logger.info("Uncollectable objects: %d" % num_uncollectable)
        if MEMPROFILE is True and j > 100:
            from pympler import muppy, summary
            objs = muppy.get_objects()
            sum = summary.summarize(objs)
            summary.print_(sum)
            sys.exit()


def main(epochs, noaction=False, restore=False):
    if not noaction:
        model = build_model()
        if restore:
            model = load_weights(model, S.DMG_MODELSTRING)
            logger.info("Weights loaded from {} successfully.".format(S.DMG_MODELSTRING))
        model.compile(optimizer=tf.keras.optimizers.RMSprop(),#tf.keras.optimizers.Adam(lr=0.00001),
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy', score.damage_f1_score])
    else:
        model = None
    train_seq = BuildingDataflow(files=get_training_files())
    valid_seq = BuildingDataflow(files=get_validation_files())
    callback = tf.keras.callbacks.ModelCheckpoint(S.DMG_MODELSTRING.replace(".hdf5", "-best.hdf5"), save_weights_only=True, save_best_only=True)

    try:
        model.fit(train_seq, validation_data=valid_seq, epochs=epochs,
                            verbose=1, callbacks=[callback],
                            validation_steps=len(valid_seq), shuffle=False,
                            use_multiprocessing=False,
                            max_queue_size=10)
    except KeyboardInterrupt:
        model.model.save_weights(S.DMG_MODELSTRING)
        logger.info("Saved to {}".format(S.DMG_MODELSTRING))
    """
    try:
        for i in range(epochs):
            logger.info("Epoch %d of %d" % (i, epochs))
            epoch(model, train_seq, valid_seq, noaction)
    except KeyboardInterrupt:
        model.model.save_weights("motokimura-damage.hdf5")
        logger.info("Saved.")
        logger.info("Profiling heap...")
        from pympler import muppy, summary
        objs = muppy.get_objects()
        sum = summary.summarize(objs)
        summary.print_(sum)
        sys.exit()
    model.model.save_weights("motokimura-damage.hdf5")
    logger.info("Saved.")
    """

def display():
    from show import display_images
    df = flow.BuildingDataflow()
    for i in range(len(df)):
        boxes, klasses = df[i]
        show.display_images(boxes, list(map(lambda x: str(x), klasses)))
    
    sys.exit()


def cli(show: ("Just show the data that will be fed to the network", "flag", "s"),
        noaction: ("Dry-run by iterating through samples w/o passing to the net", "flag", "n"),
        restore: ("Load saved weights to continue training", "flag", "r"),
        epochs: ("Number of training epochs", "option", "e", int)=50):

    if show:
        display()
        sys.exit()
    else:
        main(epochs=epochs, noaction=noaction, restore=restore)

if __name__ == "__main__":
    plac.call(cli)
