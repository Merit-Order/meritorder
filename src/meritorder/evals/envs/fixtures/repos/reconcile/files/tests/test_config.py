import unittest

from reconcile.config import load_profile


class ProfileTests(unittest.TestCase):
    def test_base_settings_apply(self):
        self.assertEqual(load_profile("dev").bank_feed_url, "https://feed.example-bank.test/v2")

    def test_profile_overrides_base(self):
        self.assertEqual(load_profile("production").http_timeout, 20.0)

    def test_unknown_profile(self):
        with self.assertRaises(ValueError):
            load_profile("base")


if __name__ == "__main__":
    unittest.main()
