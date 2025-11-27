# multi-path paradigm
# from model.clip_multi_path import CLIP_Multi_Path
# from model.coop_multi_path import COOP_Multi_Path
from models.PMNQP import PMNQP

def get_model(config, attributes=None, classes=None, offset=None, dset=None):
    if config.model_name == 'PMNQP':
        model = PMNQP(config, dset=dset)
    else:
        raise NotImplementedError(
            "Error: Unrecognized Model Name {:s}.".format(
                config.model_name
            )
        )

    return model
