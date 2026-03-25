import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire model to a specific file
    """
    network.save(filename)


def load_model(filename):
    """
    Loads an entire model from a specific file
    """
    return K.models.load_model(filename)
