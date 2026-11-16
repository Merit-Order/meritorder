import unittest

from reconcile.retry import RetryPolicy, with_retries


class RetryPolicyTests(unittest.TestCase):
    def test_delays_grow(self):
        self.assertEqual(RetryPolicy(max_retries=3, base_delay=1.0, factor=2.0).delays(), [1.0, 2.0, 4.0])

    def test_delays_are_capped(self):
        self.assertEqual(RetryPolicy(max_retries=3, base_delay=10.0, factor=2.0, max_delay=15.0).delays(), [10.0, 15.0, 15.0])

    def test_gives_up(self):
        calls, waits = [], []

        def flaky():
            calls.append(1)
            raise TimeoutError

        with self.assertRaises(TimeoutError):
            with_retries(flaky, RetryPolicy(max_retries=2, base_delay=0.5, factor=2.0), sleep=waits.append)()
        self.assertEqual((len(calls), waits), (3, [0.5, 1.0]))


if __name__ == "__main__":
    unittest.main()
