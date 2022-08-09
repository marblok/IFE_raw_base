import numpy as np
from unittest import TestCase
import generate_excitation

class Test(TestCase):
    def test_generate_excitation(self):
        generate_excitation.generate_excitation(1000, 1, np.empty(0), np.arange(16000), 16000, 0.01, [1000, 1100, 0.1])
        self.assertTrue(True)
