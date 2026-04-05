from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestLicenseFile:
    def test_license_file_exists(self) -> None:
        assert (PROJECT_ROOT / "LICENSE").is_file(), "LICENSE file must exist at project root"

    def test_no_uppercase_license_extension(self) -> None:
        assert not (PROJECT_ROOT / "LICENSE.TXT").exists(), "LICENSE.TXT should not exist; use LICENSE instead"

    def test_license_is_apache2(self) -> None:
        content = (PROJECT_ROOT / "LICENSE").read_text()
        assert "Apache License, Version 2.0" in content
