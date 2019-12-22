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
import damage
import plac

logger = logging.getLogger(__name__)


def write_solution(path, names:list, images:list):
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


def damage_by_building_classification(path):
    """
    Generate solution .png files, classifying damage using contiguous
    regions in the segmentation model's predicted masks in order extract
    individual building polygons from pre-disaster and post-disaster images.
    """
    # load the localization (segmentation) model
    S.BATCH_SIZE = 1
    model = train.build_model(architecture=S.ARCHITECTURE, train=True)
    model = train.load_weights(model, S.MODELSTRING)#.replace(".hdf5", "-best.hdf5"))

    # load the damage classification model
    dmg_model = damage.build_model()
    dmg_model = damage.load_weights(dmg_model, S.DMG_MODELSTRING)

    # get a dataflow for the test files
    df = flow.Dataflow(files=flow.get_test_files(), transform=False,
                       shuffle=False, buildings_only=False, batch_size=1,
                       return_stacked=True)
    i = 0
    pbar = tqdm.tqdm(total=len(df))
    # x = pre-disaster image, y = post-disaster image
    for stacked, filename in df:
        filename = os.path.basename(filename)
        x = stacked
        #filename = os.path.basename(df.samples[i][0].img_name)
        filename = filename.replace("pre", "localization").replace(".png", "_prediction.png") 
        #if os.path.exists(os.path.join("solution", filename)):
        #    continue

        # localization (segmentation)
        pred = model.predict(x)
        mask = infer.convert_prediction(pred)
        write_solution(names=[filename], images=[mask], path=path)

        # damage classification
        filename = filename.replace("localization", "damage")
        pre, post = stacked[...,:3], stacked[...,3:]#df.samples[i][0].image(), df.samples[i][1].image()
        boxes, coords = flow.Building.get_all_in(pre, post, mask)
        if len(boxes) > 0:
            labels = dmg_model.predict(boxes)
            for k, c in enumerate(coords):
                x,y,w,h = c
                mask[y:y+h,x:x+w] = np.argmax(labels[k])

        write_solution(names=[filename], images=[mask], path=path)
        pbar.update(1)
        i += 1


def damage_by_segmentation(path):
    """
    Generate solution .png files, using a single multiclass segmentation
    model to do so.
    """
    model = train.build_model(train=False)
    model = train.load_weights(model, S.MODELSTRING)
    df = flow.Dataflow(files=flow.get_test_files(), transform=False,
                       batch_size=1, buildings_only=False, shuffle=False,
                       return_postmask=False, return_stacked=False,
                       return_average=True)
    pbar = tqdm.tqdm(total=len(df))

    for image,filename in df:
        filename = os.path.basename(filename)
        filename = filename.replace("pre", "localization").replace(".png", "_prediction.png") 
        #if os.path.exists(os.path.join("solution", filename)):
        #    continue

        # localization (segmentation)
        pred = model.predict([image])
        mask = infer.convert_prediction(pred)
        write_solution(names=[filename], images=[mask], path=path)
        
        filename = filename.replace("localization", "damage")
        write_solution(names=[filename], images=[mask], path=path)

        pbar.update(1)


def damage_random(path):
    """
    Generate solution .png files using random damage.
    """
    model = train.build_model(train=False, save_path="motokimura-stacked-2-best.hdf5")
    model = train.load_weights(model, S.MODELSTRING)
    df = flow.Dataflow(files=flow.get_test_files(), transform=False,
                       batch_size=1, buildings_only=False, shuffle=False,
                       return_postmask=False, return_stacked=True,
                       return_average=False)
    pbar = tqdm.tqdm(total=len(df))

    for image,filename in df:
        filename = os.path.basename(filename)
        filename = filename.replace("pre", "localization").replace(".png", "_prediction.png") 
        #if os.path.exists(os.path.join("solution", filename)):
        #    continue

        # localization (segmentation)
        pred = model.predict([image])
        mask = infer.convert_prediction(pred)
        write_solution(names=[filename], images=[mask], path=path)
       
        mask = randomize_damage(mask)
        filename = filename.replace("localization", "damage")
        write_solution(names=[filename], images=[mask], path=path)

        pbar.update(1)


def cli(outdir: "The path to write solutions to",
        segmentation: ("Use a multiclass segmentation model", "flag", "s"),
        building: ("Use a binary segmentation model and individual building classifier", "flag", "b"),
        random: ("Randomize damage", "flag", "r")):

    if segmentation:
        damage_by_segmentation(outdir)
    elif building:
        damage_by_building_classification(outdir)
    elif random:
        damage_random(outdir)
    else:
        logger.error()


if __name__ == '__main__':
    plac.call(cli)
