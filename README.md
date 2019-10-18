# spacenet5

The only interesting thing here -- in my opinion, anyway, as there are much better sources than me for the models/theories -- is test-nonnet.py.  It uses sknw to skeletonize the ground truth masks, instead of skeletonizing the mask that a model predicted.  It scores very poorly on the T -> P (true graph injected into the predicted graph), but very well on P -> T (predicted injected into ground truth).  The APLS metric penalizes missing road sections so heavily that it almost scores higher to throw random noise at it (an exaggeration, but not much of one).
