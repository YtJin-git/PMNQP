# multi-path paradigm
# from model.clip_multi_path import CLIP_Multi_Path
# from model.coop_multi_path import COOP_Multi_Path
from models.PMNQP import PMNQP

def get_model(config, attributes=None, classes=None, offset=None, dset=None):
    if config.model_name == 'ThreeBranch':
        model = ThreeBranchPromptModel(config, dset=dset)
    elif config.model_name == 'CompBranch':
        model = CompositionPromptModel(config, dset=dset)
    elif config.model_name == 'MNSingleA':
        model = MNSingleA(config, dset=dset)
    elif config.model_name == 'PNPSingleA':
        model = PNPSingleA(config, dset=dset)
    elif config.model_name == 'cluspro':
        model = cluspro(config, attributes=attributes, classes=classes, offset=offset)
    elif config.model_name == 'DualOT':
        model = DualOT(config, dset=dset)
    elif config.model_name == 'ExtendThreeBranch':
        model = ExtendThreeBranch(config, dset=dset)
    elif config.model_name == 'ExtendThreeBranchWithPatchPrototypes':
        model = ExtendThreeBranchWithPatchPrototypes(config, dset=dset)
    elif config.model_name == 'ExtendThreeBranchWithMemory':
        model = ExtendThreeBranchWithMemory(config, dset=dset)
    elif config.model_name == 'ExtendThreeBranchWithOHEM':
        model = ExtendThreeBranchWithOHEM(config, dset=dset)
    elif config.model_name == 'ExtendThreeBranchTaylorCE':
        model = ExtendThreeBranchTaylorCE(config, dset=dset)
    elif config.model_name == 'ExtendThreeBranchWithMoEAdapter':
        model = ExtendThreeBranchWithMoEAdapter(config, dset=dset)
    elif config.model_name == 'ExtendThreeBranchWithPatchPrototypesMoEAdapter':
        model = ExtendThreeBranchWithPatchPrototypesMoEAdapter(config, dset=dset)
    elif config.model_name == 'ExtendThreeBranchWithHyperNetworkPrototypes':
        model = ExtendThreeBranchWithHyperNetworkPrototypes(config, dset=dset)
    elif config.model_name == 'ExtendThreeBranchWithHyperNetworkMoEAdapter':
        model = ExtendThreeBranchWithHyperNetworkMoEAdapter(config, dset=dset)
    else:
        raise NotImplementedError(
            "Error: Unrecognized Model Name {:s}.".format(
                config.model_name
            )
        )

    return model
