#!/usr/bin/env python3
"""
Crop Analysis AI Server - Render Deployment Ready
"""
import uvicorn
import os
import sys
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main server startup function"""
    try:
        # Get port from environment (Render sets this)
        port = int(os.environ.get("PORT", 8000))
        
        # Log startup information
        logger.info("Starting Crop Analysis AI Server...")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Port: {port}")
        logger.info(f"Environment: {os.environ.get('RENDER', 'local')}")
        
        # Start server
        uvicorn.run(
            "api_server:app",
            host="0.0.0.0",
            port=port,
            reload=False,
            log_level="info",
            access_log=True
        )
        
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()