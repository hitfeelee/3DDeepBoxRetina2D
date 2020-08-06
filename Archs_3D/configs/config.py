from Archs_3D import Register


def get_model_config(name):
    return Register.Config[name]()