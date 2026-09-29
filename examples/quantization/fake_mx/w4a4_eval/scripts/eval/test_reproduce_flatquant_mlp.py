"""CPU-only checks: python -m unittest discover -s <this directory>."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import reproduce_flatquant_mlp as runner


class ReproductionTest(unittest.TestCase):
    def test_view_preserves_model_and_leaves_config_to_serve(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            model = root / "model"
            model.mkdir()
            original = {
                "config.json": "{}",
                "weights.safetensors": "weights",
                "quant_model_description.json": "original config",
                "flatquant_params.safetensors": "original sidecar",
            }
            for name, content in original.items():
                (model / name).write_text(content)
            param = root / "archive.safetensors"
            param.write_text("archived sidecar")
            # Avoid requiring Windows symlink privileges; verify link targets separately.
            with patch.object(Path, "symlink_to", autospec=True) as link:
                view = runner.prepare_model_view(model, root / "out", param)
            targets = {call.args[1] for call in link.call_args_list}
            self.assertEqual(targets, {model / "config.json", model / "weights.safetensors"})
            self.assertFalse((view / "quant_model_description.json").exists())
            self.assertEqual((view / "flatquant_params.safetensors").read_text(), "archived sidecar")
            for name, content in original.items():
                self.assertEqual((model / name).read_text(), content)

    def test_rejects_nested_and_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            model = root / "model"
            model.mkdir()
            with self.assertRaises(ValueError):
                runner.prepare_model_view(model, model / "out", root / "unused")
            self.assertFalse((model / "out").exists())
            out = root / "out"
            out.mkdir()
            with self.assertRaises(FileExistsError):
                runner.prepare_model_view(model, out, root / "unused")

    def test_invalid_arguments_fail_before_loading_dependencies(self):
        for args in (["--datasets", "math500", "math500"], ["--card", "-1"], ["--port", "65536"]):
            with self.subTest(args=args), patch("sys.argv", ["runner", "--model", "model", "--out", "out", *args]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
                    runner.main()
                self.assertEqual(result.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
