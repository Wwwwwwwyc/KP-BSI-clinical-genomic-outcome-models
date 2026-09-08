from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "03_models" / "clinical_genomic_models"
SCRIPT_DIR = MODEL_DIR / "scripts"
SUPPLEMENTARY_SCRIPT = ROOT / "04_analysis" / "supplementary_statistics" / "scripts" / "build_supplementary_model_statistics.py"


class PublicCodeContractTests(unittest.TestCase):
    def test_published_entrypoints_exist(self) -> None:
        expected = [
            SCRIPT_DIR / "modeling_contract.py",
            SCRIPT_DIR / "build_nested_information_models.py",
            SCRIPT_DIR / "compare_model_families.py",
            SCRIPT_DIR / "build_compact_model.py",
            SUPPLEMENTARY_SCRIPT,
        ]
        missing = [str(path.relative_to(ROOT)) for path in expected if not path.exists()]
        self.assertEqual(missing, [])

    def test_python_entrypoints_print_help_without_private_data(self) -> None:
        env = {**os.environ, "PYTHONUTF8": "1"}
        for script_name in [
            "build_nested_information_models.py",
            "compare_model_families.py",
            "build_compact_model.py",
        ]:
            with self.subTest(script=script_name):
                result = subprocess.run(
                    [sys.executable, str(SCRIPT_DIR / script_name)],
                    cwd=ROOT,
                    env=env,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                for section in ["Function", "Required Arguments", "Optional Arguments", "Output"]:
                    self.assertIn(section, result.stdout)
                self.assertIn("--run", result.stdout)

        result = subprocess.run(
            [sys.executable, str(SUPPLEMENTARY_SCRIPT)],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--run", result.stdout)

    def test_modeling_contract_imports(self) -> None:
        spec = importlib.util.spec_from_file_location("modeling_contract", SCRIPT_DIR / "modeling_contract.py")
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        self.assertEqual(module.VALIDATION_SPLIT, "validation_2022_2023")
        self.assertEqual(module.TEST_SPLITS, ["test_2024_2025"])
        self.assertEqual(module.resolve_gwas_panels(), {})
        with self.assertRaises(ValueError):
            module.outcome_tiers("death_30d")


if __name__ == "__main__":
    unittest.main()
