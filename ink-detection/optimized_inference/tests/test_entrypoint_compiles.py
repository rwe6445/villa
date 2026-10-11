import py_compile
import unittest
from pathlib import Path

ENTRYPOINT = Path(__file__).resolve().parents[1] / "entrypoint.py"


class EntrypointCompilesTests(unittest.TestCase):
    def test_entrypoint_is_valid_python(self):
        # entrypoint.py is the Docker ENTRYPOINT; a stray unindented import inside
        # run_inference_step (#1812) made the container exit with IndentationError
        # before doing anything.
        py_compile.compile(str(ENTRYPOINT), doraise=True)


if __name__ == "__main__":
    unittest.main()
