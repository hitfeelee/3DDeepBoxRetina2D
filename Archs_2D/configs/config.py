from Archs_2D import Register
from Archs_2D.configs import mobilev2_retina_fpn_configs
from Archs_2D.configs import mobilev3_retina_fpn_configs
from Archs_2D.configs import shufflenet_retina_fpn_configs

def get_model_config(name):
    return Register.Config[name]()