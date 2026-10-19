import os
import pytest
from annotation_loader import AnnotationLoader

# Path to the extracted annotation CSVs (already extracted by previous steps)
BASE_DIR = os.path.join('e:', os.sep, 'LSE_TRANSLATOR', 'data', 'external', 'LSE-Health-UVigo', 'SWL-LSE', 'annotations_extracted', 'ANNOTATIONS')

def test_loader_loads_and_validates():
    loader = AnnotationLoader(BASE_DIR)
    # Should not raise any validation errors
    loader.validate()
    stats = loader.summary()
    # Verify row counts match the known numbers
    assert stats['train']['rows'] == 6348
    assert stats['val']['rows'] == 1052
    assert stats['test']['rows'] == 600
    # Verify no duplicate recording IDs within each split
    for split in ['train', 'val', 'test']:
        assert stats[split]['duplicate_recordings'] == 0
    # Verify overall class coverage is 300
    found, expected = loader.class_coverage()
    assert found == expected == 300
