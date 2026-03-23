import os

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.webp')


def make_dataset(directory: str):
    """Recursively collect all image file paths under *directory*.

    Args:
        directory (str): root directory to search.

    Returns:
        list[tuple[str, str]]: list of ``(filename_without_ext, full_path)``
                               pairs, sorted by filename.
    """
    if not os.path.isdir(directory):
        raise NotADirectoryError(f'Not a directory: {directory}')

    results = []
    for root, _, files in os.walk(directory):
        for fname in sorted(files):
            if any(fname.lower().endswith(ext) for ext in IMG_EXTENSIONS):
                full_path = os.path.join(root, fname)
                stem      = os.path.splitext(fname)[0]
                results.append((stem, full_path))

    return results
