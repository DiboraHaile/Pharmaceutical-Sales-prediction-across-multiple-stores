import unittest
import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join('./scripts')))
import modeling

df_train = pd.read_csv("data/train.csv")

columns = ['created_at', 'source', 'original_text','clean_text', 'sentiment','polarity','subjectivity', 'lang', 'favorite_count', 'retweet_count', 
    'original_author', 'screen_count', 'followers_count','friends_count','possibly_sensitive', 'hashtags', 'user_mentions', 'place', 'place_coord_boundaries']

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
        self.assertEqual(modeling.get_features(df).columns.tolist(),["Store","Sales","Customers","Open","Promo","StateHoliday","SchoolHoliday","Year","Part of the month","IsWeekend"])

    

if __name__ == '__main__':
	unittest.main()