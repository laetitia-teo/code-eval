import hydra
from omegaconf import OmegaConf

from train import train
from eval import eval


@hydra.main(config_name='high_fitness', config_path='conf', version_base='1.2')
def main(config):
    if config.run_id is None:
        run_id = 'test'
    else:
        run_id = config.run_id

    trainer = None
    if config.train:
        trainer = train(config, run_id=run_id)

    if config.eval:
        eval(config, run_id=run_id, trainer=trainer)


if __name__ == "__main__":
    main()
