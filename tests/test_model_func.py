import unittest
import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join('./scripts')))
import modeling

path = 'data/train.csv'
version='trainV1'

df_train,_ = modeling.load_versions(path,version)

class TestModelingFunc(unittest.TestCase):
    """
		A class for unit-testing function in the modelling.py file

		Args:
        -----
			unittest.TestCase this allows the new class to inherit
			from the unittest module
	"""

    def setUp(self) -> pd.DataFrame:
        self.df = df_train
        # tweet_df = self.df.get_tweet_df()         


    # def handle_outliers(self):
    #     self.assertEqual(modeling.handle_outliers, )

    def test_isweekend(self):
        self.assertEqual(modeling.isweekend(7),1)

    def test_format_datetime(self):
        self.assertIsInstance(modeling.format_datetime(df_train)["Date"][0],pd._libs.tslibs.timestamps.Timestamp)

    def test_get_features(self):
        df = modeling.format_datetime(self.df)
        self.assertEqual(modeling.get_features(df).columns.tolist(),["Store","IsHoliday","IsWeekend","Promo","Year","Part of the month"])

    

if __name__ == '__main__':
	unittest.main()