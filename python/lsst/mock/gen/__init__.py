"""LSST Science Pipelines adapters for the mock image generator.

This package wraps ``mock_lsst_image_generation`` (a standalone, non-LSST
package) in Science Pipelines data products: afw exposures, a butler
repository and dataset types, and a ``MockCoaddGenTask`` PipelineTask. It
imports the stack, and it imports ``mock_lsst_image_generation``, so both must
be available: set up a stack and put ``mock_lsst_image_generation`` on
``PYTHONPATH`` (see the README).
"""
