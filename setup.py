"""
Setup and Installation Script for WFC AI Integration System
Luke's RAG Implementation with Multi-Agent Architecture

This script helps set up the complete system including:
- Python environment and dependencies
- RAG system initialization  
- Database connections
- System validation
"""

import asyncio
import subprocess
import sys
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemSetup:
    """Setup and validation for WFC AI Integration System"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.requirements_file = self.base_dir / "requirements.txt"
    
    def check_python_version(self):
        """Check Python version compatibility"""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print("❌ Python 3.8 or higher is required")
            return False
        
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    
    def install_dependencies(self):
        """Install required Python packages"""
        try:
            print("📦 Installing dependencies...")
            
            if not self.requirements_file.exists():
                print("❌ requirements.txt not found")
                return False
            
            # Install requirements
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file)
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Failed to install dependencies: {result.stderr}")
                return False
            
            print("✅ Dependencies installed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error installing dependencies: {e}")
            return False
    
    def check_data_files(self):
        """Check if required data files exist"""
        data_dir = self.base_dir / "data"
        
        required_files = [
            "nodes_facilityzones.csv",
            "nodes_machines.csv", 
            "nodes_operators.csv",
            "nodes_sensors.csv",
            "nodes_workorders.csv",
            "nodes_maintenancelogs.csv",
            "nodes_productionbatches.csv",
            "rels_has_log.csv",
            "rels_has_work_order.csv",
            "rels_located_in.csv",
            "rels_monitored_by.csv",
            "rels_operated_by.csv",
            "rels_part_of_batch.csv"
        ]
        
        missing_files = []
        for file in required_files:
            if not (data_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            print(f"⚠️ Missing data files: {', '.join(missing_files)}")
            print("   The system will work but with limited data")
        else:
            print("✅ All data files present")
        
        return len(missing_files) == 0
    
    def check_environment_variables(self):
        """Check Neo4j environment variables"""
        required_vars = ['NEO4J_URI', 'NEO4J_USER', 'NEO4J_PASSWORD']
        
        print("🔐 Checking environment variables...")
        
        for var in required_vars:
            value = os.environ.get(var)
            if value:
                print(f"✅ {var}: {'*' * min(8, len(value))}...")
            else:
                print(f"⚠️ {var}: Using default value")
        
        return True
    
    async def test_rag_system(self):
        """Test RAG system initialization"""
        try:
            print("🔍 Testing RAG system...")
            
            # Import and test RAG system
            from document_processor import RAGSystem
            
            rag = RAGSystem()
            rag.initialize()
            
            # Test a simple query
            result = rag.query("test query", k=1)
            
            if result['num_results'] >= 0:  # Accept 0 results for empty data
                print("✅ RAG system working")
                return True
            else:
                print("❌ RAG system returned invalid results")
                return False
                
        except Exception as e:
            print(f"❌ RAG system test failed: {e}")
            return False
    
    async def test_graph_connection(self):
        """Test graph database connection"""
        try:
            print("📊 Testing graph database connection...")
            
            from database_agent import GraphAgent
            
            agent = GraphAgent()
            success = await agent.initialize()
            
            if success:
                # Test with a simple query
                test_result = await agent.test_connection()
                if test_result['success']:
                    print("✅ Graph database connection working")
                    return True
                else:
                    print(f"❌ Graph database test failed: {test_result.get('error')}")
                    return False
            else:
                print("❌ Graph agent initialization failed")
                return False
                
        except Exception as e:
            print(f"❌ Graph database test failed: {e}")
            return False
    
    async def run_system_validation(self):
        """Run complete system validation"""
        print("\n🧪 Running system validation...")
        
        try:
            from ai_system import WFCIntegrationSystem
            
            system = WFCIntegrationSystem()
            success = await system.initialize()
            
            if success:
                print("✅ Complete system validation passed")
                
                # Show status
                await system.show_system_status()
                
                # Cleanup
                await system.cleanup()
                return True
            else:
                print("❌ System validation failed")
                return False
                
        except Exception as e:
            print(f"❌ System validation error: {e}")
            return False
    
    async def run_setup(self):
        """Run complete setup process"""
        print("🚀 WFC AI Integration System Setup")
        print("=" * 50)
        
        steps = [
            ("Python Version", self.check_python_version),
            ("Dependencies", self.install_dependencies),
            ("Data Files", self.check_data_files),
            ("Environment", self.check_environment_variables),
            ("RAG System", self.test_rag_system),
            ("Graph Database", self.test_graph_connection),
            ("System Validation", self.run_system_validation)
        ]
        
        results = {}
        
        for step_name, step_func in steps:
            print(f"\n📋 Step: {step_name}")
            print("-" * 30)
            
            try:
                if asyncio.iscoroutinefunction(step_func):
                    result = await step_func()
                else:
                    result = step_func()
                
                results[step_name] = result
                
            except Exception as e:
                print(f"❌ {step_name} failed: {e}")
                results[step_name] = False
        
        # Summary
        print(f"\n{'=' * 50}")
        print("🎯 SETUP SUMMARY")
        print("=" * 50)
        
        passed = 0
        for step_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{step_name:20} {status}")
            if result:
                passed += 1
        
        print(f"\nResults: {passed}/{len(results)} steps passed")
        
        if passed == len(results):
            print("\n🎉 Setup completed successfully!")
            print("🚀 You can now run: python interactive_query.py")
        else:
            print(f"\n⚠️ Setup completed with {len(results) - passed} issues")
            print("🔧 Please resolve the failed steps before running the system")
        
        return passed == len(results)

def create_env_template():
    """Create environment variable template"""
    env_template = """# WFC AI Integration Environment Variables
# Copy this to .env and fill in your values

# Neo4j Database Configuration
NEO4J_URI=neo4j+s://62f9b154.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=U32P3onr7idgSWbqklVReZQ8BVRH_BWH3_A5Oj83oq0

# RAG System Configuration (optional)
RAG_MODEL_NAME=all-MiniLM-L6-v2
RAG_CACHE_DIR=./rag_cache

# Logging
LOG_LEVEL=INFO
"""
    
    env_file = Path(".env.template")
    env_file.write_text(env_template)
    print(f"✅ Created {env_file} - copy to .env and customize")

async def main():
    """Main setup entry point"""
    setup = SystemSetup()
    
    print("🏭 WFC AI Integration System Setup")
    print("   Luke's RAG Implementation with Multi-Agent Architecture")
    print()
    
    # Create environment template
    create_env_template()
    
    # Run setup
    try:
        success = await setup.run_setup()
        
        if success:
            print("\n✨ Next Steps:")
            print("1. Customize .env file if needed")
            print("2. Run: python interactive_query.py (recommended)")
            print("3. Or run: python integration_system.py (direct system)")
        else:
            print("\n🔧 Troubleshooting:")
            print("1. Check error messages above")
            print("2. Ensure Neo4j credentials are correct")
            print("3. Verify all CSV data files are present")
            print("4. Try: pip install -r requirements.txt")
    
    except KeyboardInterrupt:
        print("\n👋 Setup interrupted")
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())