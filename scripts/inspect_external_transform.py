"""Find the actual transform used by external MyDataset."""
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, r'D:\Projects\Series-Photo-Selection')

# Import the dataloader module to see what transform is imported
import data.dataloader as dl_module

logger.info(f"Module attributes: {[x for x in dir(dl_module) if not x.startswith('_')]}")

# Check if transform is defined
if hasattr(dl_module, 'transform'):
    transform = dl_module.transform
    logger.info(f"\nFound transform: {transform}")
    logger.info(f"Transform type: {type(transform)}")
    
    # Show the transform function source
    import inspect
    transform_source = inspect.getsource(transform)
    logger.info(f"\nTransform function source:\n{transform_source}")
    
    # Check FixScaleCrop
    if hasattr(dl_module, 'FixScaleCrop'):
        logger.info(f"\nFixScaleCrop source:\n{inspect.getsource(dl_module.FixScaleCrop)}")
else:
    logger.info("No 'transform' attribute found in dataloader module")

# Try to find where transform is imported from
import inspect
source = inspect.getsource(dl_module)
transform_imports = [line for line in source.split('\n') if 'transform' in line.lower() and ('import' in line or 'from' in line)]
logger.info(f"\nImport lines with 'transform': {transform_imports}")
