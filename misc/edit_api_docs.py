"""
Helper script to edit API documentation files.

This helper script modifies the output of `sphinx-apidoc` to be a bit more readable in
a cross-platform way. This script:

1. Removes the `modules.rst` file if it exists.
2. Copies the content of `vaxflux.rst` to `index.rst`, removing the first two lines
   which are typically the module name and a brief description.
"""

from pathlib import Path
from shutil import copyfileobj


def main() -> None:
    """Main function to edit API documentation files."""
    api = Path.cwd() / "docs" / "api"
    if not api.exists():
        return None
    modules_rst = api / "modules.rst"
    if modules_rst.exists():
        modules_rst.unlink()
    vaxflux_rst = api / "vaxflux.rst"
    index_rst = api / "index.rst"
    if vaxflux_rst.exists():
        with vaxflux_rst.open("r") as from_file, index_rst.open("w") as to_file:
            from_file.readline()
            from_file.readline()
            to_file.writelines(["API Reference\n", "=============\n"])
            copyfileobj(from_file, to_file)
        vaxflux_rst.unlink()


if __name__ == "__main__":
    main()
