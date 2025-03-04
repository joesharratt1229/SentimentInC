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
    
    
        

class TestCalculateDiscreteBinProportions(unittest.TestCase):
    def setUp(self):
        # Create sample data for tests
        self.sample_data = pd.DataFrame({
            'category': ['A', 'B', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B'],
            'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        
        # Sample unique values
        self.unique_values = ['A', 'B', 'C']
        
        # Sample training proportions matching the number of unique values
        self.training_proportions = [0.3, 0.4, 0.3]
    
    def test_normal_execution(self):
        """Test normal execution with valid inputs."""
        test_proportions, training_props = calculate_discrete_bin_proportions(
            self.sample_data, 'category', self.unique_values, self.training_proportions
        )
        
        # Check output types
        self.assertIsInstance(test_proportions, list)
        self.assertIsInstance(training_props, list)
        
        # Check lengths match
        self.assertEqual(len(test_proportions), len(training_props))
        self.assertEqual(len(test_proportions), len(self.unique_values))
        
        # Check sum of proportions is close to 1
        self.assertAlmostEqual(sum(test_proportions), 1.0, places=6)
    
    def test_empty_dataframe(self):
        """Test with empty dataframe."""
        empty_df = pd.DataFrame(columns=['category'])
        
        with self.assertRaises(Exception):
            calculate_discrete_bin_proportions(
                empty_df, 'category', self.unique_values, self.training_proportions
            )
    
    def test_missing_values(self):
        """Test with missing values in the column."""
        df_with_nulls = self.sample_data.copy()
        df_with_nulls.loc[2, 'category'] = None
        df_with_nulls.loc[5, 'category'] = np.nan
        
        test_proportions, _ = calculate_discrete_bin_proportions(
            df_with_nulls, 'category', self.unique_values, self.training_proportions
        )
        
        # Check that we still get valid proportions (sum to 1)
        self.assertAlmostEqual(sum(test_proportions), 1.0, places=6)
        
        # Expected to have 8 values instead of 10
        self.assertEqual(len(df_with_nulls[df_with_nulls['category'].notnull()]), 8)
    
    def test_mismatched_unique_values_and_proportions(self):
        """Test mismatched length between unique_values and training_proportions."""
        wrong_proportions = [0.5, 0.5]  # One fewer than needed
        
        with self.assertRaises(AssertionError):
            calculate_discrete_bin_proportions(
                self.sample_data, 'category', self.unique_values, wrong_proportions
            )
    
    def test_column_not_in_dataframe(self):
        """Test with a column that doesn't exist in the dataframe."""
        with self.assertRaises(Exception):
            calculate_discrete_bin_proportions(
                self.sample_data, 'non_existent_column', self.unique_values, self.training_proportions
            )
    ]
)
