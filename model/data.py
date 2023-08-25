from keras.preprocessing.image import ImageDataGenerator
from model.batch_stabilization import stabilize_batch_size, is_converging
from model.settings import TRAINING_PATH, VALIDATION_PATH, TESTING_PATH, HEIGHT, WIDTH

def flow_from_directory(batch_size, history, n_last_epochs):
    training_data_generator = ImageDataGenerator(rescale = 1./255,
                              rotation_range = 359,
                              shear_range = 0.2,
                              width_shift_range = 0.2,
                              height_shift_range = 0.2,
                              zoom_range = 0.2,
                              horizontal_flip = True,
                              vertical_flip = True,
                              preprocessing_function = None)

    validation_data_generator = ImageDataGenerator(rescale = 1./255)
    test_data_generator = ImageDataGenerator(rescale = 1./255)

    converging = is_converging(history = history, n_last_epochs = n_last_epochs)
    new_batch_size = stabilize_batch_size(converging = converging, batch_size = batch_size)

    training_generator = training_data_generator.flow_from_directory(TRAINING_PATH,
                                                                 target_size = (HEIGHT, WIDTH),
                                                                 batch_size = new_batch_size,
                                                                 class_mode = "categorical",
                                                                 color_mode = 'grayscale',
                                                                 )

    validation_generator = validation_data_generator.flow_from_directory(VALIDATION_PATH,
                                                                        target_size = (HEIGHT, WIDTH),
                                                                        batch_size = 1,
                                                                        class_mode = "categorical",
                                                                        color_mode = 'grayscale',
                                                                        )

    test_generator = test_data_generator.flow_from_directory(TESTING_PATH,
                                                            target_size = (HEIGHT, WIDTH),
                                                            batch_size = 1,
                                                            class_mode = "categorical",
                                                            color_mode = 'grayscale',
                                                            )
    
    return training_generator, validation_generator, test_generator, new_batch_size