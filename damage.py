import numpy as np
import logging
import infer
from flow import Dataflow
import train
import deeplabmodel
import settings as S
import tensorflow as tf
import plac
import unet
import sys
import tensorflow as tf

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


def build_model(backbone=S.ARCHITECTURE,
                train=False,
                classes=4):

    model = unet.MotokimuraUnet(classes=2)
    model = model.convert_to_damage_classifier()
    return model


def load_weights(model, save_file):
    model.model.load_weights(save_file)
    return model


def extract_patches(pre, post, mask, return_masks=False, max_x=S.DAMAGE_MAX_X, max_y=S.DAMAGE_MAX_Y):
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
    """
    preboxes = []
    postboxes = []
    klasses = []
    masks = []
    # extract individual buildings from the mask
    rectangles = infer.bounding_rectangles(mask)
    for rect in rectangles:
        # store a view for each building rectangle (region of interest) that was found
        x,y = rect
        if x.stop-x.start <= 5 or y.stop-y.start <= 5:
            continue
        if x.stop-x.start >= max_x:
            x = slice(x.start, x.start + max_x - 1)
        if y.stop-y.start >= max_y:
            y = slice(y.start, y.start + max_y - 1)

        prebox = pre[x.start:x.stop,y.start:y.stop]
        postbox = post[x.start:x.stop,y.start:y.stop]
        retmask = mask[x.start:x.stop,y.start:y.stop]

        klass = mask[x.start:x.stop,y.start:y.stop]
        klass = mode(klass[np.nonzero(klass)])
        if klass > 5 or klass < 0:
            logger.warning("Unrecognized class! Setting to appropriate value")
            klass = 1
        preboxes.append(prebox)#.astype(np.uint8))
        postboxes.append(postbox)#.astype(np.uint8))
        masks.append(retmask)
        klass_one_hot = [0] * 5
        klass_one_hot[klass-1] = 1
        klasses.append(klass_one_hot)

    # return all the (pre-disaster, post-disaster) ROIs, their corresponding damage classes,
    # and optionally the mask we used.
    # If extraction algorithm failed, return empty values
    if return_masks is True:
        if len(preboxes) < 1:
            return (np.empty(0),np.empty(0)),[],np.empty(0)
        return (np.array(preboxes), np.array(postboxes)), klasses, np.array(masks)
    else:
        if len(preboxes) < 1:
            return (np.empty(0),np.empty(0)),[]
        return (np.array(preboxes), np.array(postboxes)), klasses


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


class BuildingDataflow(tf.keras.utils.Sequence):
    """
    A dataflow to train on building damage classification.  Expects buildings to exist 
    in the `topdir` passed to __init__(), and to be .png format files of pre- and post-
    disaster images laid out on top of each other.  Each pre- and post-disaster image
    pair has multiple building images that use the pre-disaster name only for the directory
    in which to find the buildings.  The target class for a given image is in the filename,
    e.g. `topdir/hurricane-harvey-pre-00001/0:1.png` is the first building in hurricane
    harvey image 00001, and the buildings in the file belong to the 'minor-damage' class.
    """
    def __init__(self, topdir="data/buildings", batch_size=32):
        self.files = []
        self.topdir = topdir
        self.batch_size = batch_size

        dirs = os.listdir(topdir)
        for dir in dirs:
            files = os.listdir(dir)
            for file in files:
                self.files.append(os.path.absname(file))

    def __len__(self):
        length = int(np.ceil(len(self.files) / float(self.batch_size)))
        return len(self.files)

    def __getitem__(self, idx):
        for filename in self.files[idx*self.batch_size:(idx+1)*self.batch_size]:
            img = skimage.io.imread(filename)
            klass = int(filename.split(":")[-1][0])

            x.append(img)
            y.append(klass)

        return np.array(x), np.array(y)


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
    from flow import get_training_files, get_validation_files
    if not noaction:
        model = build_model()
        if restore:
            model = load_weights(model, "motokimura-damage.hdf5")
            logger.info("Weights loaded successfully.")
        model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='categorical_crossentropy')
    else:
        model = None
    train_seq = DamageDataflow(files=get_training_files(), shuffle=False, transform=0.3)
    valid_seq = DamageDataflow(files=get_validation_files(), shuffle=False, transform=False)

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


def display():
    from show import display_images
    df = DamageDataflow(return_masks=True, batch_size=1)
    for (xs,ys), klasses, masks, buildings in df:#[df[2152], df[2153], df[2154], df[2155]]:
        images = []
        names = []
        for i in range(len(xs)):
            images.append(xs[i])
            images.append(ys[i])
            images.append(masks[i])
            images.append(buildings[i])
            names.append(klasses[i])
            names.append(klasses[i])
            names.append(klasses[i])
            names.append(klasses[i])
        display_images(images, names)


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
