#!/usr/bin/env python3
"""
Quick endpoint test script - run with: python scripts/test_endpoints.py
"""
import socket
import urllib.request
import urllib.error
import json
import ssl

BASE_URL = 'http://localhost'
OLLAMA_URL = 'http://localhost:11434'
CHROMADB_URL = 'http://localhost:8020'
TIMEOUT = 30

def make_request(url, method='GET', data=None, headers=None):
    """Make HTTP request using urllib."""
    if headers is None:
        headers = {}
    if data:
        data = json.dumps(data).encode('utf-8')
        headers['Content-Type'] = 'application/json'
    
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=TIMEOUT, context=ctx) as response:
            return response.status, json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception as e:
        return None, str(e)

def check_port(host, port):
    """Check if a port is open."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

print('=' * 60)
print('  LIVE ENDPOINT INTEGRATION TEST RESULTS')
print('=' * 60)
print()

tests = []

# Container/Service Health
print('📦 CONTAINER/SERVICE HEALTH')
print('-' * 40)

# Nginx
status, data = make_request(f'{BASE_URL}/')
if status == 200:
    print(f'✅ Nginx Gateway: OK')
    tests.append(('Nginx', True))
else:
    print(f'❌ Nginx Gateway: FAILED - {data}')
    tests.append(('Nginx', False))

# Ollama
status, data = make_request(f'{OLLAMA_URL}/api/tags')
if status == 200 and data:
    models = [m['name'] for m in data.get('models', [])]
    print(f'✅ Ollama: OK - Models: {models}')
    tests.append(('Ollama', True))
else:
    print(f'❌ Ollama: FAILED - {data}')
    tests.append(('Ollama', False))

# ChromaDB
status, data = make_request(f'{CHROMADB_URL}/api/v1/heartbeat')
if status == 200:
    print(f'✅ ChromaDB: OK')
    tests.append(('ChromaDB', True))
else:
    print(f'❌ ChromaDB: FAILED - {data}')
    tests.append(('ChromaDB', False))

# Redis
if check_port('localhost', 6379):
    print(f'✅ Redis: OK (port 6379)')
    tests.append(('Redis', True))
else:
    print(f'❌ Redis: FAILED - Port not open')
    tests.append(('Redis', False))

# MySQL
if check_port('localhost', 3307):
    print(f'✅ MySQL: OK (port 3307)')
    tests.append(('MySQL', True))
else:
    print(f'❌ MySQL: FAILED - Port not open')
    tests.append(('MySQL', False))

print()

# Authentication
print('🔐 AUTHENTICATION')
print('-' * 40)

token = None
status, data = make_request(f'{BASE_URL}/auth/login', 'POST', {'username': 'admin', 'password': 'admin123'})
if status == 200 and data:
    token = data.get('access_token')
    print(f'✅ POST /auth/login: OK')
    tests.append(('Login', True))
else:
    print(f'❌ POST /auth/login: {status} - {data}')
    tests.append(('Login', False))

if token:
    headers = {'Authorization': f'Bearer {token}'}
    status, data = make_request(f'{BASE_URL}/auth/me', headers=headers)
    if status == 200 and data:
        print(f'✅ GET /auth/me: OK - User: {data.get("username")}')
        tests.append(('Auth Me', True))
    else:
        print(f'❌ GET /auth/me: {status}')
        tests.append(('Auth Me', False))

print()

# Status Endpoints
print('📊 STATUS ENDPOINTS')
print('-' * 40)

status_endpoints = [
    ('/status', 'System Status'),
    ('/health', 'Health Check'),
    ('/ingest/status', 'Ingest Status'),
    ('/memory/status', 'Memory Status'),
]

for endpoint, name in status_endpoints:
    status, data = make_request(f'{BASE_URL}{endpoint}')
    if status == 200:
        extra = ''
        if endpoint == '/status' and data:
            extra = f' - Model: {data.get("model", "?")}'
        print(f'✅ GET {endpoint}: OK{extra}')
        tests.append((name, True))
    else:
        print(f'❌ GET {endpoint}: {status}')
        tests.append((name, False))

print()

# Protected Endpoints
if token:
    print('🔒 PROTECTED ENDPOINTS')
    print('-' * 40)
    
    protected_endpoints = [
        ('/conversations', 'Conversations'),
        ('/clients', 'Clients'),
        ('/documents', 'Documents'),
        ('/documents/stats', 'Doc Stats'),
        ('/documents/formats', 'Doc Formats'),
        ('/models', 'Models List'),
        ('/models/current', 'Current Model'),
        ('/models/providers', 'Providers'),
        ('/evaluation/datasets', 'Eval Datasets'),
        ('/evaluation/runs', 'Eval Runs'),
        ('/admin/stats', 'Admin Stats'),
        ('/admin/config', 'Admin Config'),
        ('/admin/users', 'Admin Users'),
    ]
    
    for endpoint, name in protected_endpoints:
        status, data = make_request(f'{BASE_URL}{endpoint}', headers=headers)
        if status and status < 400:
            print(f'✅ GET {endpoint}: OK')
            tests.append((name, True))
        else:
            print(f'❌ GET {endpoint}: {status}')
            tests.append((name, False))
    
    print()
    
    # Test document search
    print('🔍 SEARCH & WRITE OPERATIONS')
    print('-' * 40)
    
    status, data = make_request(f'{BASE_URL}/documents/search', 'POST', {'query': 'test', 'top_k': 3}, headers)
    if status == 200:
        results = data.get('results', []) if data else []
        print(f'✅ POST /documents/search: OK - {len(results)} results')
        tests.append(('Doc Search', True))
    else:
        print(f'❌ POST /documents/search: {status}')
        tests.append(('Doc Search', False))
    
    # Test create conversation
    status, data = make_request(f'{BASE_URL}/conversations', 'POST', {'title': 'Test Conv'}, headers)
    if status == 200 and data:
        conv_id = data.get('id', '')
        print(f'✅ POST /conversations: OK - Created: {conv_id[:8]}...')
        tests.append(('Create Conv', True))
    else:
        print(f'❌ POST /conversations: {status}')
        tests.append(('Create Conv', False))

print()
print('=' * 60)
passed = sum(1 for _, ok in tests if ok)
failed = sum(1 for _, ok in tests if not ok)
print(f'  SUMMARY: {passed} passed, {failed} failed')
print('=' * 60)

if failed > 0:
    print()
    print('Failed tests:')
    for name, ok in tests:
        if not ok:
            print(f'  ❌ {name}')
