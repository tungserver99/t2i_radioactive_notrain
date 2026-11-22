import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


class RunLoggingTests(unittest.TestCase):
    def test_tee_run_log_captures_stdout_and_stderr_without_hiding_console(self):
        from ogsc.progress import tee_run_log

        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / 'logs' / 'run.log'
            console_out, console_err = io.StringIO(), io.StringIO()
            with redirect_stdout(console_out), redirect_stderr(console_err):
                with tee_run_log(log_path, 'bitwise-validation'):
                    print('sample progress: 25/100 | ETA 3.0 min')
                    print('sample warning', file=__import__('sys').stderr)

            contents = log_path.read_text(encoding='utf-8')

        self.assertIn('sample progress: 25/100 | ETA 3.0 min', contents)
        self.assertIn('sample warning', contents)
        self.assertIn('sample progress: 25/100 | ETA 3.0 min', console_out.getvalue())
        self.assertIn('sample warning', console_err.getvalue())


if __name__ == '__main__':
    unittest.main()
