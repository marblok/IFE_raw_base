from vowel_synthesis.vowels_synthesizer import test
from utils.log_config import LogConfig

# import numpy as np

# sq1 = np.random.SeedSequence()
# rng_state = sq1.entropy

# sequence_index = 10

# sq1 = np.random.SeedSequence(rng_state)
# rng_seed = sq1.generate_state(sequence_index+1)
# rng = np.random.default_rng(rng_seed[-1])
# x = rng.random(5)
# print(x)

# sequence_index = 10
# sq1 = np.random.SeedSequence(rng_state)
# rng_seed = sq1.generate_state(sequence_index+1)
# rng = np.random.default_rng(rng_seed[-1])
# x = rng.random(5)
# print(x)

log_cfg = LogConfig()
log_cfg.init_logging("vowel_synthesis_test")

test(1000, "rng_state.entropy")
#test(1000) # test(1000, None)
print("Test finished")