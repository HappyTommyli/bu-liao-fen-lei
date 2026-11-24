"""Historical TCN configs remain covered after the desktop defaults to prefix CERA."""
from cloth_trainer.modern_config import ModernConfig
from cloth_trainer.tcn_models import default_spec
from cloth_trainer.tcn_ui import TCNTrainerWindow as RealWindow


def legacy_focused_config():
    return ModernConfig(architectures=('tcn',),signals=('by',),tcn_spec=dict(default_spec(128),pooling='mean_std'),cv_folds=1)


def TCNTrainerWindow(*args,**kwargs):
    w=RealWindow(*args,**kwargs);w.config=legacy_focused_config();w.sync_controls();return w
