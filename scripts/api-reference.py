"""Generate API reference documentation for vaxflux."""

from pathlib import Path

from ruamel.yaml import YAML


def get_module_name(file_path: Path, src_root: Path) -> str:
    """Convert a file path to a Python module name.

    Args:
        file_path: Path to the Python file.
        src_root: Root source directory.

    Returns:
        The fully qualified module name.
    """
    parts = list(file_path.relative_to(src_root).parts)

    # Remove .py extension from the last part
    parts[-1] = parts[-1].removesuffix(".py")

    # Remove __init__ from module names
    if parts[-1] == "__init__":
        parts = parts[:-1]

    return ".".join(parts)


def is_public_module(module_path: Path) -> bool:
    """Check if a module is public (not prefixed with _).

    Args:
        module_path: Path to the module file.

    Returns:
        `True` if the module is public, `False` otherwise.
    """
    return not any(
        (part.startswith("_") or part == "__init__.py") for part in module_path.parts
    )


def get_python_files(package_dir: Path) -> list[Path]:
    """Get all Python module files in a package directory.

    Args:
        package_dir: Path to the package directory.

    Returns:
        List of Python file paths, excluding '__init__.py'.
    """
    return [file for file in sorted(package_dir.glob("*.py")) if is_public_module(file)]


def generate_api_doc(item_path: Path, src_root: Path, output_dir: Path) -> str:
    """Generate API documentation for a package or module.

    Args:
        item_path: Path to the package directory or module file.
        src_root: Root source directory.
        output_dir: Directory to write the documentation file.

    Returns:
        The name of the generated documentation file (without .md extension).
    """
    # Get the module name from the path
    module_name = get_module_name(item_path, src_root)

    # Create output file name (last part of module name)
    doc_name = module_name.split(".")[-1]
    output_file = output_dir / f"{doc_name}.md"

    # Start building the content
    lines = [
        f"# {doc_name.capitalize()}",
        "",
        f"::: {module_name}",
        "",
    ]

    # Write the file
    output_file.write_text("\n".join(lines))

    return doc_name


def generate_package_doc(package_name: str, output_dir: Path) -> str:
    """Generate API documentation for the main package.

    Args:
        package_name: The name of the package (e.g., 'vaxflux').
        output_dir: Directory to write the documentation file.

    Returns:
        The name of the generated documentation file (without .md extension).
    """
    doc_name = package_name
    output_file = output_dir / f"{doc_name}.md"

    # Start building the content
    lines = [
        f"# {doc_name.capitalize()}",
        "",
        f"::: {package_name}",
        "",
    ]

    # Write the file
    output_file.write_text("\n".join(lines))

    return doc_name


def update_mkdocs_nav(mkdocs_path: Path, api_docs: list[str], package_doc: str) -> None:
    """Update the mkdocs.yml navigation with generated API docs.

    Uses ruamel.yaml to preserve comments and formatting.

    Args:
        mkdocs_path: Path to the mkdocs.yml file.
        api_docs: List of API document names (without .md extension).
        package_doc: The package documentation name (without .md extension).
    """
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096
    yaml.explicit_start = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    with mkdocs_path.open("r", encoding="utf-8") as f:
        config = yaml.load(f)

    # Find the Reference -> API section in navigation
    nav = config.get("nav", [])
    reference_idx = None
    for i, item in enumerate(nav):
        if isinstance(item, dict) and "API Reference" in item:
            reference_idx = i
            break

    # If not found, exit
    if reference_idx is None:
        return

    # Build the new API navigation structure
    # Start with the package documentation, then add module docs
    api_nav = [{package_doc.capitalize(): f"api/{package_doc}.md"}]
    api_nav.extend({doc.capitalize(): f"api/{doc}.md"} for doc in sorted(api_docs))

    nav[reference_idx]["API Reference"] = api_nav
    with mkdocs_path.open("w", encoding="utf-8") as f:
        yaml.dump(config, f)


def main() -> None:
    """Generate API reference documentation for all packages."""
    # Define paths
    repo_root = Path(__file__).parent.parent
    src_root = repo_root / "src"
    vaxflux_root = src_root / "vaxflux"
    output_dir = repo_root / "docs" / "api"
    mkdocs_path = repo_root / "mkdocs.yml"

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean up existing markdown files in the API reference directory
    for md_file in output_dir.glob("*.md"):
        md_file.unlink()

    # Generate main package documentation
    package_doc = generate_package_doc("vaxflux", output_dir)

    # Find all public modules
    generated_docs = []
    for item in sorted(vaxflux_root.iterdir()):
        if not is_public_module(item):
            continue
        if item.is_file() and item.suffix == ".py":
            doc_name = generate_api_doc(item, src_root, output_dir)
            generated_docs.append(doc_name)

    # Update mkdocs.yml navigation
    update_mkdocs_nav(mkdocs_path, generated_docs, package_doc)


if __name__ == "__main__":
    main()
