#!/usr/bin/env python3
"""
Demonstration of CVRPLIB Processing Pipeline

This script demonstrates how to use the CVRPLIB processing pipeline
to convert .vrp files to HDF5 format for training.

Usage:
    python demo_cvrplib_pipeline.py
"""

import logging
from pathlib import Path
from cvrplib_processor import CVRPLIBProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Demonstrate the CVRPLIB processing pipeline."""

    logger.info("🚀 CVRPLIB Processing Pipeline Demo")
    logger.info("=" * 50)

    # Define paths
    input_folder = Path(__file__).parent / "test_data" / "sample_cvrplib"
    output_file = Path(__file__).parent / "test_data" / "demo_output.h5"

    # Check if test data exists
    if not input_folder.exists():
        logger.error(f"❌ Test data directory not found: {input_folder}")
        logger.info("Please run the test suite first to download sample data:")
        logger.info("  python test_cvrplib_pipeline.py")
        return

    # Create processor
    logger.info(f"📁 Input folder: {input_folder}")
    logger.info(f"📄 Output file: {output_file}")

    processor = CVRPLIBProcessor(str(input_folder), str(output_file))

    # Process all instances
    logger.info("\n🔄 Processing CVRPLIB instances...")
    processor.process_all()

    # Display results
    logger.info(f"\n📊 Results:")
    logger.info(f"  ✅ Successfully processed: {processor.processed_count}")
    logger.info(f"  ❌ Failed: {processor.failed_count}")
    logger.info(f"  📄 Output file: {output_file}")

    # Demonstrate force regeneration if solutions exist
    sol_files = list(input_folder.glob("*.sol"))
    if sol_files:
        logger.info(f"\n🔄 Demonstrating force regeneration...")
        force_output_file = Path(__file__).parent / "test_data" / "demo_force_output.h5"
        processor_force = CVRPLIBProcessor(str(input_folder), str(force_output_file), force_regenerate=True)
        processor_force.process_all()

        logger.info(f"  🔄 Force regeneration completed")
        logger.info(f"  📄 Force regeneration output: {force_output_file}")

        # Check for backup files
        backup_files = list(input_folder.glob("*_backup_*.sol"))
        logger.info(f"  💾 Backup files created: {len(backup_files)}")

    if processor.processed_count > 0:
        # Convert to relative path from neo-lrp-GT directory
        relative_output_path = f"../training_data_sampling/{output_file.name}"

        logger.info(f"\n🎯 Next steps:")
        logger.info(f"  1. Update neo-lrp-GT/train.py to use: '{relative_output_path}'")
        logger.info(f"  2. Run training: cd ../neo-lrp-GT && python train.py")

        # Show example code
        logger.info(f"\n💡 Example train.py modification:")
        logger.info(f"  # Replace the data loading line in neo-lrp-GT/train.py:")
        logger.info(f"  train_data, test_data, _ = prepare_pretrain_data(")
        logger.info(f"      '{relative_output_path}',")
        logger.info(f"      split_ratios=[0.8, 0.2, 0.0],")
        logger.info(f"  )")

        # Show additional usage options
        logger.info(f"\n🔄 Additional usage options:")
        logger.info(f"  # Force regenerate solutions (with backup):")
        logger.info(f"  python cvrplib_processor.py /path/to/instances output.h5 --force-regenerate")
        logger.info(f"")
        logger.info(f"  # Set custom time limit (in seconds):")
        logger.info(f"  python cvrplib_processor.py /path/to/instances output.h5 --time-limit 600")
        logger.info(f"")
        logger.info(f"  # Combine options:")
        logger.info(f"  python cvrplib_processor.py /path/to/instances output.h5 --force-regenerate --time-limit 60")

    logger.info(f"\n✨ Demo complete!")


if __name__ == "__main__":
    main()