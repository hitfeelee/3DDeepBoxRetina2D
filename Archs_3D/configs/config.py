from Archs_3D import Register
from Archs_3D.configs.mobilev2_retina3d_configs import *
from Archs_3D.configs.mobilev2_deepbox3d_configs import *

def get_model_config(name):
    return Register.Config[name]()