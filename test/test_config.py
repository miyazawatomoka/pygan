from config import Config

class TestConfig:
    def test_constant(self):
        assert(not Config.ndf is None)