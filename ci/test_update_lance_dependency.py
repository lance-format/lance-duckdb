from __future__ import annotations

import unittest

from ci.update_lance_dependency import (
    LANCE_CRATES,
    remove_lance_patch_entries,
    update_lance_versions_in_dependencies,
)


class UpdateLanceDependencyTest(unittest.TestCase):
    def test_updates_normal_and_dev_dependencies(self) -> None:
        normal_crates = [crate for crate in LANCE_CRATES if crate != "lance-file"]
        lines = ["[dependencies]"]
        for crate in normal_crates:
            lines.append(f'{crate} = "13.0.0"')
        lines.extend(
            [
                "",
                "[dev-dependencies]",
                'lance-file = "13.0.0"',
                'lance-io = { version = "13.0.0", features = ["test-util"] }',
            ]
        )

        updated = update_lance_versions_in_dependencies(lines, "14.0.1")

        self.assertEqual(updated, set(LANCE_CRATES))
        self.assertIn('lance-file = "14.0.1"', lines)
        self.assertIn(
            'lance-io = { version = "14.0.1", features = ["test-util"] }',
            lines,
        )

    def test_removes_all_lance_patch_entries(self) -> None:
        lines = [
            "[patch.crates-io]",
            'lance-datafusion = { git = "https://example.com/lance" }',
            'lance-io = { git = "https://example.com/lance" }',
            'unrelated = { git = "https://example.com/unrelated" }',
        ]

        remove_lance_patch_entries(lines)

        self.assertEqual(
            lines,
            [
                "[patch.crates-io]",
                "",
                "",
                'unrelated = { git = "https://example.com/unrelated" }',
            ],
        )


if __name__ == "__main__":
    unittest.main()
