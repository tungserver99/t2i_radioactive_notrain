import io
import time
import unittest
from contextlib import redirect_stdout


class ProgressTests(unittest.TestCase):
    def test_progress_line_reports_fraction_elapsed_and_eta(self):
        from ogsc.progress import progress_line

        message = progress_line('manifest hashing', 250, 1000, 120)

        self.assertIn('250/1000', message)
        self.assertIn('25%', message)
        self.assertIn('2.0 min elapsed', message)
        self.assertIn('ETA 6.0 min', message)

    def test_stage_context_reports_start_completion_and_duration(self):
        from ogsc.progress import stage_progress

        output = io.StringIO()
        with redirect_stdout(output):
            with stage_progress('teacher scoring'):
                pass

        self.assertIn('teacher scoring: started', output.getvalue())
        self.assertIn('teacher scoring: completed in', output.getvalue())

    def test_long_stage_emits_a_heartbeat_while_work_is_running(self):
        from ogsc.progress import stage_progress

        output = io.StringIO()
        with redirect_stdout(output):
            with stage_progress('FID/KID', interval_seconds=0.01):
                time.sleep(0.025)

        self.assertIn('FID/KID: still running', output.getvalue())


if __name__ == '__main__':
    unittest.main()
