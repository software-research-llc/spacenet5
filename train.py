import snflow as flow
import keras
import keras.backend as K
import numpy as np
import snmodel
from keras.applications import xception
import time
import tensorflow as tf
import loss
import unet
from tensorflow_examples.models.pix2pix import pix2pix

#tf.compat.v1.disable_eager_execution()

EPOCHS = 25
model = None
model_file = "model.tf-2"
i = 0
iters = 0

@tf.function
def train_step(input_image, target):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = unet.generator(input_image, training=True)

    disc_real_output = unet.discriminator([input_image, target], training=True)
    disc_generated_output = unet.discriminator([input_image, gen_output], training=True)

    gen_loss = unet.generator_loss(disc_generated_output, gen_output, target)
    disc_loss = unet.discriminator_loss(disc_real_output, disc_generated_output)

  unet.generator_gradients = gen_tape.gradient(gen_loss,
                                          unet.generator.trainable_variables)
  unet.discriminator_gradients = disc_tape.gradient(disc_loss,
                                               unet.discriminator.trainable_variables)

  unet.generator_optimizer.apply_gradients(zip(unet.generator_gradients,
                                          unet.generator.trainable_variables))
  unet.discriminator_optimizer.apply_gradients(zip(unet.discriminator_gradients,
                                              unet.discriminator.trainable_variables))

def fit(train_ds, epochs, test_ds):
  for epoch in range(epochs):
    # Train
    for input_image, target in train_ds:
      train_step(input_image, target)

    # Test on the same image so that the progress of the model can be 
    # easily seen.
    for example_input, example_target in test_ds.take(1):
      generate_images(unet.generator, example_input, example_target)

def train_gan(model, seq, epochs=EPOCHS):
    global iters
    if isinstance(model, pix2pix.Pix2pix):
        """
        start = time.time()
        totaltime = time.time() - start
        print("{:15s} {:^20s} {:^20s} {:^15s} {:^20s}".format("Full epochs", "Secs per iteration", "Mins per full epoch", "Batch size", "Samples remaining"))
        print("{:15s} {:^20.2f} {:^20.2f} {:^15s} {:^20s}".format(str(i), steptime / len(seq), steptime, str(len(x)), str(len(seq) * len(x) - iters * len(x))))
        iters += 1
        """
        model.epochs = 1
        start = time.time()
        history = model.train(seq, model_file)
        stop = time.time()
        steptime = stop - start
        totaltime = len(seq) * steptime / 60
        iters += 1
        print("{:15s} {:^20s} {:^20s} {:^15s} {:^20s}".format("Full epochs", "Secs per iteration", "Mins per full epoch", "Batch size", "Samples remaining"))
        print("{:15s} {:^20.2f} {:^20.2f} {:^15s} {:^20s}".format(str(i),        steptime,           totaltime,            str(seq.batch_size), "0"))
        print(history)

def train(model, seq, epochs=EPOCHS):
    global iters
    for x, y in seq:
        start = time.time()
        if seq.batch_size == 1:
            history = model.fit(x, y, epochs=epochs, verbose=1, use_multiprocessing=True)
        else:
            history = model.fit(x, y, batch_size=seq.batch_size, epochs=epochs, validation_split=0.1, verbose=1, use_multiprocessing=True)
        stop = time.time()
        steptime = stop - start
        totaltime = len(seq) * steptime / 60
        iters += 1
        print("{:15s} {:^20s} {:^20s} {:^15s} {:^20s}".format("Full epochs", "Secs per iteration", "Mins per full epoch", "Batch size", "Samples remaining"))
        print("{:15s} {:^20.2f} {:^20.2f} {:^15s} {:^20s}".format(str(i), steptime, totaltime, str(len(x)), str(len(seq) * len(x) - iters * len(x))))
    print(history.history['loss'])

def custom_accuracy(y_true, y_pred):
    """Return the percentage of pixels that were correctly predicted as belonging to a road"""
    tp = tf.math.count_nonzero(y_true * y_pred)
    total_pos = tf.math.count_nonzero(y_true)
    return tf.dtypes.cast(tp, dtype=tf.float64) / tf.maximum(tf.constant(1, dtype=tf.float64), tf.dtypes.cast(total_pos, dtype=tf.float64))

def custom_loss(y_true, y_pred):
    tp = tf.math.count_nonzero(y_true * y_pred)
    total_pos = tf.math.count_nonzero(y_true)
    loss = tf.constant(1, dtype=tf.float64) - tf.dtypes.cast(tp, dtype=tf.float64) / tf.maximum(tf.constant(1, dtype=tf.float64), tf.dtypes.cast(total_pos, dtype=tf.float64))
    return loss

def main():
    global model
    global i
    if model is None:
        model = snmodel.build_model()
        print("WARNING: starting from a new model")
        time.sleep(5)
    if not isinstance(model, pix2pix.Pix2pix):
        snmodel.compile_model(model)
        model.summary()
    seq = flow.SpacenetSequence.all(model=model)
    i = 0
    while True:
        if isinstance(model, pix2pix.Pix2pix):
            train_gan(model, seq)
        else:
            train(model, seq)
        i += 1
        print("Loops through training data: %d" % i)

if __name__ == '__main__':
    try:
        model = snmodel.load_model()
        model.checkpoint.restore(model_file)
    except Exception as exc:
        print(exc)
    try:
        main()
    except KeyboardInterrupt:
        print("Finished %d full epochs." % i)
        print("\nSaving to file in 5...")
        time.sleep(5)
        print("\nSaving...")
        try:
            snmodel.save_model(model, model_file)
        except Exception as exc:
            print(exc)
            model.checkpoint.save(model_file)
