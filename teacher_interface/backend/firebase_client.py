"""
Firebase Client Authentication
Tests connection to MIT Firebase server using JWT authentication
"""

import requests
import json
from typing import Optional, Dict, Any

class FirebaseClientAuth:
    """Client for authenticating with Firebase server using JWT"""
    
    def __init__(self, server_url: str, username: str, password: str):
        """Initialize client with server credentials"""
        self.server_url = server_url.rstrip('/')
        self.username = username
        self.password = password
        self.jwt_token = None
    
    def authenticate(self) -> bool:
        """
        Authenticate with server and get JWT token
        Returns True if successful, False otherwise
        """
        try:
            auth_data = {
                "username": self.username,
                "password": self.password
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            print(f"Authenticating with server: {self.server_url}")
            response = requests.post(
                f"{self.server_url}/auth", 
                data=json.dumps(auth_data), 
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                self.jwt_token = result.get('access_token')
                print("JWT authentication successful")
                return True
            else:
                print(f"Authentication failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"Authentication error: {e}")
            return False
    
    def make_authenticated_request(self, endpoint: str, method: str = "GET", 
                                 params: Optional[Dict] = None, 
                                 data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Make authenticated request to server endpoint
        Returns JSON response or None if failed
        """
        if not self.jwt_token:
            print("No JWT token available. Please authenticate first.")
            return None
        
        try:
            headers = {
                "Accept": "application/json",
                "Authorization": f"JWT {self.jwt_token}"
            }
            
            url = f"{self.server_url}{endpoint}"
            
            if method.upper() == "GET":
                response = requests.get(url, params=params, headers=headers)
            elif method.upper() == "POST":
                response = requests.post(url, data=json.dumps(data), headers=headers)
            else:
                print(f"Unsupported HTTP method: {method}")
                return None
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Request failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Request error: {e}")
            return None
    
    def test_top_level_node_access(self, top_level_node: str) -> bool:
        """
        Test access to top-level node
        Returns True if successful, False otherwise
        """
        print(f"\nTesting access to top-level node: {top_level_node}")
        
        # Try to get nodes from the top-level node
        params = {
            "subject_id": top_level_node,
            "path": "/"
        }
        
        result = self.make_authenticated_request("/get_nodes", params=params)
        
        if result is not None:
            print(f"Successfully accessed top-level node: {top_level_node}")
            print(f"Data keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            return True
        else:
            print(f"Failed to access top-level node: {top_level_node}")
            return False
    
    def get_available_endpoints(self) -> Dict[str, Any]:
        """Get information about available endpoints"""
        endpoints = {
            "whoami": self.make_authenticated_request("/whoami"),
            "get_storybooks": self.make_authenticated_request("/get_storybooks"),
            "get_students": self.make_authenticated_request("/get_students")
        }
        
        return {k: v for k, v in endpoints.items() if v is not None}

def main():
    """Test Firebase client authentication"""
    
    # Configuration - you'll need to update these with your actual credentials
    SERVER_URL = "https://prg-webhost.media.mit.edu"
    USERNAME = "your_username_here"  # Update this
    PASSWORD = "your_password_here"   # Update this
    TOP_LEVEL_NODE = "storybook-list-by-school"  # Update this if different
    
    print("Firebase Client Authentication Test")
    print("=" * 50)
    
    # Create client
    client = FirebaseClientAuth(SERVER_URL, USERNAME, PASSWORD)
    
    # Test authentication
    if client.authenticate():
        print("\nAuthentication successful!")
        
        # Test whoami endpoint
        print("\nTesting whoami endpoint...")
        whoami = client.make_authenticated_request("/whoami")
        if whoami:
            print(f"Current user: {whoami}")
        
        # Test top-level node access
        client.test_top_level_node_access(TOP_LEVEL_NODE)
        
        # Test other endpoints
        print("\nTesting other endpoints...")
        endpoints = client.get_available_endpoints()
        for endpoint, data in endpoints.items():
            print(f"{endpoint}: {type(data)} - {len(data) if isinstance(data, (dict, list)) else 'N/A'} items")
        
    else:
        print("\nAuthentication failed!")
        print("Please check your credentials and server URL.")

if __name__ == "__main__":
    main()
