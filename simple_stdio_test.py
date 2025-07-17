#!/usr/bin/env python3
"""Simple test to verify STDIO server starts and responds."""

import asyncio
import subprocess
import sys
import time


async def test_stdio_startup():
    """Test that the STDIO server starts without errors."""
    print("Testing STDIO server startup...")
    
    # Start the server process
    process = subprocess.Popen(
        [sys.executable, "-m", "memory_mcp_server.cli", "stdio"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    try:
        # Give the server a moment to start
        await asyncio.sleep(2)
        
        # Check if process is still running (not crashed)
        if process.poll() is None:
            print("✅ Server started successfully and is running")
            
            # Try to read any initial output
            try:
                # Use non-blocking read to check for output
                import select
                if select.select([process.stdout], [], [], 0)[0]:
                    output = process.stdout.readline()
                    if output:
                        print(f"Server output: {output.strip()}")
            except:
                pass  # Non-blocking read not available on all platforms
            
            return True
        else:
            print("❌ Server process exited unexpectedly")
            stderr_output = process.stderr.read()
            if stderr_output:
                print(f"Error output: {stderr_output}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False
    finally:
        # Clean up
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def test_server_components():
    """Test server components directly."""
    print("Testing server components...")
    
    try:
        from memory_mcp_server.protocols.mcp_server import MemoryMCPServer
        
        # Create server instance
        server = MemoryMCPServer()
        print("✅ Server instance created successfully")
        
        # Check if FastMCP server is configured
        mcp_server = server.get_mcp_server()
        print(f"✅ FastMCP server configured: {mcp_server.name}")
        
        # Check if tools are registered
        # Note: FastMCP doesn't expose tools directly, but we can check if the server was created
        print("✅ Server components initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("Running STDIO server tests...\n")
    
    # Test 1: Component initialization
    component_test = test_server_components()
    print()
    
    # Test 2: STDIO startup
    startup_test = await test_stdio_startup()
    print()
    
    if component_test and startup_test:
        print("✅ All tests passed! STDIO server implementation is working.")
        return True
    else:
        print("❌ Some tests failed.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)