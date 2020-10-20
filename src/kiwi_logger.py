from typing import Union, Any

from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import LightningLoggerBase
import kiwi


class KiwiLogger(LightningLoggerBase):

    @property
    def experiment(self) -> Any:
        pass

    @property
    def name(self) -> str:
        return "KiwiLogger"

    @property
    def version(self) -> Union[int, str]:
        pass

    @rank_zero_only
    def log_hyperparams(self, params):
        for key, value in vars(params).items():
            kiwi.log_param(key, value)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        for k, v in metrics.items():
            kiwi.log_metric(key=k, value=v, step=step)

    def save(self):
        # Optional. Any code necessary to save logger data goes here
        pass

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass
