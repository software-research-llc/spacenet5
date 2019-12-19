import tqdm
import numpy as np
import skimage
import skimage.io
import damage
import settings as S
import sys
import os
import flow


OUTPUT_DIR = "data/buildings"


def extract_all(df, output_dir=OUTPUT_DIR):
    """
    Likely to exhaust system RAM before it finishes, but will return all individual
    buildings in a list of (preimg-slice, postimg-slice, damage-subtype) tuples.
    """
    ret = []
    for (pre, post) in tqdm.tqdm(df.samples):
        preimg = pre.image()
        postimg = post.image()
        for bldg in sample.buildings:
            prebox = bldg.extract_from_image(preimg)
            postbox = bldg.extract_from_image(postimg)
            ret.append( (prebox,postbox,bldg.color()) )

    return ret


def write_all(output_dir=OUTPUT_DIR):
    df = flow.BuildingDataflow(shuffle=False)
    filelist = list(map(lambda x: os.path.basename(x[0]), flow.get_training_files()))
    pbar = tqdm.tqdm(total=len(filelist))

    for i,buildings in enumerate(df):
        prename = filelist[i]
        prename = prename.replace(".json", "").replace(".png", "")
        dirname = os.path.join(output_dir, prename)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        for j, (prebox,postbox,klass) in enumerate(buildings):

            outfile = os.path.join(dirname, str(j) + ":" + str(klass) + ".png")
            skimage.io.imsave(outfile, comboimg, check_contrast=False)


def run(output_dir=OUTPUT_DIR):
    df = damage.DamageDataflow(files=flow.get_training_files(), shuffle=False)
    filelist = list(map(lambda x: os.path.basename(x[0]), flow.get_training_files()))
    pbar = tqdm.tqdm(total=len(filelist))

    for i in range(len(filelist)):
        try:
            prename = filelist[i]
        except Exception:
            continue
        prename = prename.replace(".json", "").replace(".png", "")
        dirname = os.path.join(output_dir, prename)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        try:
            (_, _), klasses, _, buildings = df[i]
        except Exception:
            continue
        for j in range(len(buildings)):
            try:
                b = buildings[j]
            except Exception:
                continue
            outfile = os.path.join(dirname, str(j) + ":" + str(klasses[j].index(1)) + ".png")
            if os.path.exists(outfile):
                continue
            skimage.io.imsave(outfile, b.astype(np.uint8), check_contrast=False)
        pbar.update(1)

if __name__ == "__main__":
    run()
