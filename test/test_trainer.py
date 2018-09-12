from Trainer.trainer import Trainer
class TestTrainer:
    def test_trainer_init(self):
        x = Trainer()
        assert not x.dataloader is None
