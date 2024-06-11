'''
Unit tests for the crps module.
'''
import unittest
import numpy as np
from crpsnb.crps import crps_neighborhood_chunk


class TestCRPS(unittest.TestCase):
    '''
    Unit tests for the crps module.
    '''

    def test_crps_neighborhood_chunk_returns_ndarray(self):
        '''
        Test that crps_neighborhood_chunk returns a numpy array.
        '''

        ens_chunk = np.array([[1, 2, 3], [4, 5, 6]])
        yobs_chunk = np.array([[1, 2, 3], [4, 5, 6]])
        result = crps_neighborhood_chunk(ens_chunk, yobs_chunk)
        self.assertIsInstance(result, np.ndarray)

    def test_crps_neighborhood_chunk_returns_correct_shape(self):
        '''
        Test that crps_neighborhood_chunk returns the correct shape.
        '''

        ens_chunk = np.array([[1, 2, 3], [4, 5, 6]])
        yobs_chunk = np.array([[1, 2, 3], [4, 5, 6]])
        result = crps_neighborhood_chunk(ens_chunk, yobs_chunk)
        self.assertEqual(result.shape, (2,))

    def test_crps_neighborhood_chunk_returns_correct_values(self):
        '''
        Test that crps_neighborhood_chunk returns the correct values.
        '''

        ens_chunk = np.array([[1, 2, 3], [4, 5, 6]])
        yobs_chunk = np.array([[1, 2, 3], [4, 5, 6]])
        result = crps_neighborhood_chunk(ens_chunk, yobs_chunk)
        self.assertTrue(np.allclose(result, np.array([0, 0])))

    def test_crps_neighborhood_chunk_orientation(self):
        '''
        Test that crps_neighborhood_chunk is negatively oriented.
        '''

        ens_chunk = np.array([[1, 2, 3], [4, 5, 6]])
        yobs_chunk = np.array([[1, 2, 3], [4, 5, 6]])
        result = crps_neighborhood_chunk(ens_chunk, yobs_chunk)

        ens_chunk2 = ens_chunk + 0.1
        result2 = crps_neighborhood_chunk(ens_chunk2, yobs_chunk)

        self.assertTrue(np.all(result2 > result))

    def test_crps_neighborhood_chunk_nan_handling(self):
        '''
        Test that crps_neighborhood_chunk handles NaN rows correctly.
        '''

        yobs_chunk = np.array([
            [1, 2, 3],
            [4, 5, 7],
            [np.nan, np.nan, np.nan]
        ])
        ens_chunk = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ])

        result_1 = crps_neighborhood_chunk(ens_chunk, yobs_chunk)

        yobs_chunk2 = np.array([
            [1, 2, 3],
            [4, 5, 7],
        ])
        ens_chunk2 = np.array([
            [1, 2, 3],
            [4, 5, 6],
        ])

        result_2 = crps_neighborhood_chunk(ens_chunk2, yobs_chunk2)

        self.assertTrue(np.allclose(result_1[:2], result_2))


if __name__ == '__main__':
    unittest.main()
