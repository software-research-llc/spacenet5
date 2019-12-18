"""
Generate solution files.
"""
import random
import skimage
import os
import tqdm
import settings as S
import numpy as np
import train
import flow
import infer
import logging

logger = logging.getLogger(__name__)


def write_solution(names:list, images:list, path="solution"):
    """
    Write out .png image(s).
    """
    assert len(names) == len(images)
    for i, img in enumerate(images):
        outfname = os.path.join(path, names[i])
        skimage.io.imsave(outfname, img.astype(np.uint8), check_contrast=False)


def randomize_damage(img):
    """
    Make each building pixel a random value; likelihood is proportional to that
    class' expected frequency.
    """
    assert list(img.shape) == S.SAMPLESHAPE[:2], f"expected shape (1024,1024), got {img.shape}"
    # distribute pixels among the classes proportional to the expected distributions
    top = 0
    nodamage = range(top, top + 80000); top += 80000
    minordamage = range(top, top + 11000); top += 11000
    majordamage = range(top, top + 8000); top += 8000
    destroyed = range(top, top + 8000); top += 8000

    for x,y in zip(*np.nonzero(img)):
            num = random.randint(0,top-1)
            if num in nodamage:
                pix_val = 1
            elif num in minordamage:
                pix_val = 2
            elif num in majordamage:
                pix_val = 3
            elif num in destroyed:
                pix_val = 4
            else:
                raise Exception("result %d is outside known values" % num)
            img[x][y] = pix_val
    return img


if __name__ == '__main__':
    # load the localization (segmentation) model
    model = train.build_model(train=False)
    model = train.load_weights(model)

    # load the damage classification model
    dmg_model = damage.build_model()
    dmg_model = damage.load_weights(dmg_model)

    # get a dataflow for the test files
    df = flow.Dataflow(files=flow.get_test_files(), transform=False, shuffle=False, buildings_only=False)

    i = 0
    pbar = tqdm.tqdm(total=len(df))
    # x = pre-disaster image, y = post-disaster image
    for x, y in df:
        filename = os.path.basename(df.samples[i][0].img_name)
        filename = filename.replace("pre", "localization").replace(".png", "_prediction.png") 
        if os.path.exists(os.path.join("solution", filename)):
            continue

        # localization (segmentation)
        pred = model.predict(x)
        mask = infer.weave_pred(pred)
        write_solution(names=[filename], images=[mask])

        # damage classification
        filename = filename.replace("localization", "damage")
        dmgdict = damage.extract_patches(x, y, mask, return_dict=True)

        # FIXME: do something smart with un-classified
        for k, (x,y) in enumerate(dmgdict['bbox']):
            # set all "1" pixels from the localization step to the predicted damage class
            klass = dmgdict['class'][k]
            box = mask[x.start:x.stop,y.start:y.stop,:]
            # klass is one-hot encoded and ranges from 0 to 4
            box[box > 0] = np.argmax(klass) + 1

        write_solution(names=[filename], images=[mask])

        pbar.update(1)
        i += 1