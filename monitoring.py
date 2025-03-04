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


from test_spark_utils import SparkTestCase
import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from extract_metadata import subset_df_by_column_type, get_bin_stats, process_categorical_feature

class TestExtractMetadata(SparkTestCase):
    
    def setUp(self):
        # Create sample input DataFrame
        input_data = [
            ("user1", 25, 100.5, "category_A", True),
            ("user2", 30, 150.2, "category_B", False),
            ("user3", 22, 90.7, "category_A", True),
            ("user4", 45, 200.1, "category_C", False),
            ("user5", 33, 120.8, "category_B", True)
        ]
        input_schema = ["user_id", "age", "value", "category", "flag"]
        self.input_df = self.spark.createDataFrame(input_data, schema=input_schema)
        
        # Create sample data_type_df (column metadata)
        type_data = [
            ("age", "numerical", "Age in years"),
            ("value", "numerical", "Value in dollars"),
            ("category", "categorical", "Category type"),
            ("flag", "categorical", "Boolean flag")
        ]
        type_schema = ["DescriptionName", "Type", "Description"]
        self.data_type_df = self.spark.createDataFrame(type_data, schema=type_schema)
    
    def test_subset_df_by_column_type_numerical(self):
        """Test extracting numerical columns."""
        subset_df, columns = subset_df_by_column_type(
            self.input_df, self.data_type_df, "numerical"
        )
        
        # Check returned columns list
        self.assertEqual(set(columns), {"age", "value"})
        
        # Check DataFrame has only requested columns
        self.assertEqual(set(subset_df.columns), {"age", "value"})
        
        # Check row count remains the same
        self.assertEqual(subset_df.count(), self.input_df.count())
    
    def test_subset_df_by_column_type_categorical(self):
        """Test extracting categorical columns."""
        subset_df, columns = subset_df_by_column_type(
            self.input_df, self.data_type_df, "categorical"
        )
        
        # Check returned columns list
        self.assertEqual(set(columns), {"category", "flag"})
        
        # Check DataFrame has only requested columns
        self.assertEqual(set(subset_df.columns), {"category", "flag"})
    
    def test_subset_df_nonexistent_columns(self):
        """Test with columns in metadata that don't exist in the input DataFrame."""
        # Create metadata with a column that doesn't exist in input_df
        extra_type_data = [
            ("age", "numerical", "Age in years"),
            ("nonexistent_col", "numerical", "This column doesn't exist")
        ]
        extra_type_df = self.spark.createDataFrame(extra_type_data, ["DescriptionName", "Type", "Description"])
        
        subset_df, columns = subset_df_by_column_type(
            self.input_df, extra_type_df, "numerical"
        )
        
        # Should only return columns that actually exist
        self.assertEqual(set(columns), {"age"})
    
    def test_get_bin_stats(self):
        """Test bin statistics calculation for numerical columns."""
        edges, proportions = get_bin_stats(self.input_df, "age", 3)
        
        # Check return types
        self.assertIsInstance(edges, list)
        self.assertIsInstance(proportions, list)
        
        # Should have n_bins edges (or n_bins+1 for histogram edges)
        self.assertGreaterEqual(len(edges), 3)
        
        # Should have n_bins proportions
        self.assertEqual(len(proportions), 3)
        
        # Proportions should sum to 1
        self.assertAlmostEqual(sum(proportions), 1.0, places=6)
    
    def test_get_bin_stats_with_nulls(self):
        """Test bin stats with null values."""
        # Create DataFrame with nulls
        null_data = [
            ("user1", 25), 
            ("user2", None),
            ("user3", 22),
            ("user4", 45),
            ("user5", None)
        ]
        null_df = self.spark.createDataFrame(null_data, ["user_id", "age"])
        
        edges, proportions = get_bin_stats(null_df, "age", 2)
        
        # Proportions should still sum to 1 despite nulls
        self.assertAlmostEqual(sum(proportions), 1.0, places=6)
    
    def test_process_categorical_feature(self):
        """Test processing of categorical features."""
        # Assuming implementation details of process_categorical_feature
        result = process_categorical_feature(self.input_df, "category")
        
        # The function documentation suggests it returns counts and proportions
        # Adjust these assertions based on the actual implementation
        self.assertTrue("category_A" in str(result))
        self.assertTrue("category_B" in str(result))
        self.assertTrue("category_C" in str(result))
    ]
)
