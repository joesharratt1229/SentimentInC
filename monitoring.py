import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from proportions import calculate_continuous_bin_proportions

class TestCalculateContinuousBinProportions(unittest.TestCase):
    
    def setUp(self):
        # Create sample data for tests
        self.sample_data = pd.DataFrame({
            'value': [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
            'category': ['A', 'B', 'A', 'B', 'A', 'A', 'B', 'B', 'A', 'B']
        })
        
        # Sample bin edges
        self.bin_edges = [0, 3, 6, 9, 12]
        
        # Sample training proportions matching the number of bins
        self.training_proportions = [0.25, 0.25, 0.25, 0.25]
    
    def test_normal_execution(self):
        """Test normal execution with valid inputs."""
        test_proportions, training_props = calculate_continuous_bin_proportions(
            self.sample_data, 'value', self.bin_edges, self.training_proportions
        )
        
        # Check output types
        self.assertIsInstance(test_proportions, list)
        self.assertIsInstance(training_props, list)
        
        # Check lengths match
        self.assertEqual(len(test_proportions), len(training_props))
        self.assertEqual(len(test_proportions), len(self.bin_edges) - 1)
        
        # Check sum of proportions is close to 1
        self.assertAlmostEqual(sum(test_proportions), 1.0, places=6)
    
    def test_empty_dataframe(self):
        """Test with empty dataframe."""
        empty_df = pd.DataFrame(columns=['value'])
        
        with self.assertRaises(Exception):
            calculate_continuous_bin_proportions(
                empty_df, 'value', self.bin_edges, self.training_proportions
            )
    
    def test_missing_values(self):
        """Test with missing values in the column."""
        df_with_nulls = self.sample_data.copy()
        df_with_nulls.loc[2, 'value'] = None
        df_with_nulls.loc[5, 'value'] = np.nan
        
        test_proportions, _ = calculate_continuous_bin_proportions(
            df_with_nulls, 'value', self.bin_edges, self.training_proportions
        )
        
        # Check that we still get valid proportions (sum to 1)
        self.assertAlmostEqual(sum(test_proportions), 1.0, places=6)
        
        # Expected to have 8 values instead of 10
        self.assertEqual(len(df_with_nulls[df_with_nulls['value'].notnull()]), 8)
    
    def test_mismatched_bin_edges_and_proportions(self):
        """Test mismatched length between bin_edges and training_proportions."""
        wrong_proportions = [0.3, 0.3, 0.4]  # One fewer than needed
        
        with self.assertRaises(AssertionError):
            calculate_continuous_bin_proportions(
                self.sample_data, 'value', self.bin_edges, wrong_proportions
            )
    
    def test_column_not_in_dataframe(self):
        """Test with a column that doesn't exist in the dataframe."""
        with self.assertRaises(Exception):
            calculate_continuous_bin_proportions(
                self.sample_data, 'non_existent_column', self.bin_edges, self.training_proportions
            )
    
    def test_all_values_in_one_bin(self):
        """Test when all values fall into a single bin."""
        single_bin_df = pd.DataFrame({'value': [1.5, 1.8, 2.0, 2.2, 2.5]})
        bin_edges = [1, 3, 5, 7]
        training_proportions = [0.5, 0.3, 0.2]
        
        test_proportions, _ = calculate_continuous_bin_proportions(
            single_bin_df, 'value', bin_edges, training_proportions
        )
        
        # All values should be in the first bin
        self.assertAlmostEqual(test_proportions[0], 1.0, places=6)
        self.assertAlmostEqual(test_proportions[1], 0.0, places=6)
        self.assertAlmostEqual(test_proportions[2], 0.0, places=6)
    
    @patch('proportions.F')
    def test_with_mocked_f_module(self, mock_F):
        """Test using mocked F module (likely from pyspark.sql.functions)."""
        # Setup mock behavior
        mock_col = MagicMock()
        mock_F.col.return_value = mock_col
        mock_when = MagicMock()
        mock_F.when.return_value = mock_when
        mock_otherwise = MagicMock()
        mock_when.otherwise.return_value = mock_otherwise
        
        # Create a mock DataFrame that will work with our mocked functions
        mock_df = MagicMock()
        mock_filtered_df = MagicMock()
        mock_df.filter.return_value = mock_filtered_df
        
        # Setup return values for withColumn
        mock_df.withColumn.return_value = mock_df
        mock_df.count.return_value = 10
        
        # For grouped data
        mock_grouped = MagicMock()
        mock_count = MagicMock()
        mock_df.groupBy.return_value = mock_grouped
        mock_grouped.count.return_value = mock_count
        mock_count.withColumnRenamed.return_value = mock_df
        
        # For ordering and filtering
        mock_df.orderBy.return_value = mock_df
        mock_df.filter.return_value = mock_df
        
        # For final result
        mock_df.select.return_value = mock_df
        mock_df.rdd.return_value = mock_df
        mock_df.flatMap.return_value = mock_df
        mock_df.collect.return_value = [0.25, 0.25, 0.25, 0.25]
        
        # Execute function with mocks
        result = calculate_continuous_bin_proportions(
            mock_df, 'test_column', [0, 1, 2, 3, 4], [0.25, 0.25, 0.25, 0.25]
        )
        
        # Verify the result
        self.assertEqual(result[0], [0.25, 0.25, 0.25, 0.25])
        self.assertEqual(result[1], [0.25, 0.25, 0.25, 0.25])
        
        # Verify mocks were called as expected
        mock_F.col.assert_called_with('test_column')
        mock_df.filter.assert_called()
        mock_df.withColumn.assert_called()
        mock_df.groupBy.assert_called()

    @unittest.skip("Example of how to skip tests that aren't applicable")
    def test_skipped_test(self):
        """Example of a skipped test."""
        self.fail("This test should be skipped")
    
    @classmethod
    def parametrize(cls, test_method, param_list):
        """Custom parametrize method since unittest doesn't have built-in parametrization."""
        for i, params in enumerate(param_list):
            test_name = f"{test_method.__name__}_case_{i}"
            new_test = lambda self, params=params: test_method(self, *params)
            setattr(cls, test_name, new_test)
    
    def _test_with_bin_edges(self, bin_edges, training_proportions, expected_bins):
        """Helper method for parametrized tests."""
        test_proportions, _ = calculate_continuous_bin_proportions(
            self.sample_data, 'value', bin_edges, training_proportions
        )
        self.assertEqual(len(test_proportions), expected_bins)
        self.assertAlmostEqual(sum(test_proportions), 1.0, places=6)


# Parametrize tests
TestCalculateContinuousBinProportions.parametrize(
    TestCalculateContinuousBinProportions._test_with_bin_edges,
    [
        ([0, 5, 10, 15], [0.3, 0.3, 0.4], 3),
        ([0, 2, 4, 6, 8, 10, 12], [1/6, 1/6, 1/6, 1/6, 1/6, 1/6], 6),
        ([0, 12], [1.0], 1)
    ]
)
