"""Reset database config profile to match pipeline.yaml.

Run this script to update the database config after modifying pipeline.yaml.

Usage:
    python scripts/reset_db_config.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sim_bench.api.database.models import ConfigProfile
from sim_bench.api.services.config_service import get_default_config

def reset_config():
    """Reset default config profile to match pipeline.yaml."""
    # Connect to database
    db_path = project_root / "sim_bench.db"
    if not db_path.exists():
        print(f"❌ Database not found at {db_path}")
        print("   Start the API first to create the database.")
        return False
    
    engine = create_engine(f'sqlite:///{db_path}')
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Get default profile
        profile = session.query(ConfigProfile).filter(
            ConfigProfile.name == "default"
        ).first()
        
        if not profile:
            print("❌ Default config profile not found in database")
            return False
        
        # Load new config from pipeline.yaml
        new_config = get_default_config()
        
        print("📋 Current default_pipeline in database:")
        old_pipeline = profile.config.get('default_pipeline', [])
        for i, step in enumerate(old_pipeline, 1):
            print(f"   {i}. {step}")
        
        print("\n📋 New default_pipeline from pipeline.yaml:")
        new_pipeline = new_config.get('default_pipeline', [])
        for i, step in enumerate(new_pipeline, 1):
            print(f"   {i}. {step}")
        
        # Update profile
        profile.config = new_config
        session.commit()
        
        print("\n✅ Config profile updated successfully!")
        print("   Restart the API to use the new config.")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        session.rollback()
        return False
    finally:
        session.close()

if __name__ == "__main__":
    print("=" * 60)
    print("Reset Database Config Profile")
    print("=" * 60)
    print()
    
    success = reset_config()
    sys.exit(0 if success else 1)
