import os

from lightning.pytorch import loggers


class WandbLogger(loggers.wandb.WandbLogger):
    def __init__(self, stage_dir, project, group, **kwargs) -> None:

        id = os.environ.get("SLURM_JOB_ID")
        if not (
            "SLURM_JOB_QOS" in os.environ
            and "interactive" not in os.environ["SLURM_JOB_QOS"]
            and "jupyter" not in os.environ["SLURM_JOB_QOS"]
        ):
            id = None

        name = kwargs.pop("name", None) or id

        super().__init__(name=name, save_dir=stage_dir, id=id, project=project, group=group)
