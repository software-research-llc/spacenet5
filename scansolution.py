#import snflow as flow
import numpy as np
import shapely
import re
import logging
import tqdm
from shapely.geometry import LineString
import plac

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_solution(solutionfile="solution/solution.csv"):
    with open(solutionfile) as f:
        for line in tqdm.tqdm(f.readlines()[1:]):
            check_line(line)

def check(imageid, lstr, length_m, time_s):
    logger.debug("{}: length_m={}, time_s={}, speed={}".format(imageid,
                                                               length_m,
                                                               time_s,
                                                               lstr.length / time_s))
    speed = lstr.length / time_s
    if speed > 29 or speed < 7:
        logger.info("{}: speed={} (length_m={}, time_s={})".format(imageid,
                                                                      speed,
                                                                      lstr.length,
                                                                      time_s))

def check_line(line):
    imageid, lstr, length_m, time_s = get_all(line)
    try:
        if lstr.length != 0:
            check(imageid, lstr, length_m, time_s)
    except ZeroDivisionError:
        logger.warning(("ZeroDivisionError on {}, length_m={}, " +
                       "time_s={}, line={}").format(imageid, length_m, time_s, line))
    
def get_all(line):
    lstr = get_linestring(line)
    length_m = get_length_m(line)
    time_s = get_time_s(line)
    imageid = get_imageid(line)
    return imageid, lstr, length_m, time_s

def get_linestring(line):
    if "EMPTY" in line:
        return LineString()
    linestring = re.findall("\([^)]+\)", line)[0]
    linestring = linestring.replace("(", "").replace(")", "")
    points = [x.split(" ") for x in linestring.split(", ")]
    try:
        coords = [ (float(point[0]), float(point[1])) for point in points]
    except Exception as exc:
        print(str(points))
        raise exc
#    coords = re.findall("\d+\.?\d* \d+\.?\d*", linestring)
#    coords = [(float(x.split(" ")[0]), float(x.split(" ")[1])) for x in coords]
    try:
        lstr = LineString(coords)
    except Exception as exc:
        logger.critical("{} on {}".format(exc, line))
        return LineString()

    return lstr

def get_imageid(line):
    return line.split(",")[0]

def get_time_s(line):
    return float(line.split(",")[-1])

def get_length_m(line):
    return float(line.split(",")[-2])

if __name__ == "__main__":
    plac.call(check_solution)
