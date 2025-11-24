"""Enhanced connection test with detailed error information"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import (
    ELASTICSEARCH_URL,
    ELASTICSEARCH_VERIFY_CERTS,
    ELASTICSEARCH_TIMEOUT
)

print("="*80)
print("ELASTICSEARCH CONNECTION TEST")
print("="*80)

print("\n[1] Configuration:")
print(f"  URL: {ELASTICSEARCH_URL}")
print(f"  VERIFY_CERTS: {ELASTICSEARCH_VERIFY_CERTS}")
print(f"  TIMEOUT: {ELASTICSEARCH_TIMEOUT}")

# Test 1: Direct Elasticsearch client
print("\n[2] Testing direct Elasticsearch client...")
try:
    from elasticsearch import Elasticsearch
    
    es_config = {
        'hosts': [ELASTICSEARCH_URL],
        'verify_certs': ELASTICSEARCH_VERIFY_CERTS,
        'request_timeout': ELASTICSEARCH_TIMEOUT,
        'max_retries': 3,
        'retry_on_timeout': True
    }
    
    # For Elasticsearch 8.x, disable SSL warnings when cert verification is disabled
    if not ELASTICSEARCH_VERIFY_CERTS:
        es_config['ssl_show_warn'] = False
    
    print(f"  Config: {es_config}")
    
    es = Elasticsearch(**es_config)
    
    # Try ping first
    result = es.ping()
    
    if result:
        print("  ✅ SUCCESS! Direct connection works")
        info = es.info()
        print(f"     Cluster: {info['cluster_name']}")
        print(f"     Version: {info['version']['number']}")
    else:
        print("  ❌ FAILED: ping() returned False")
        # Try to get more info by calling info() which will throw an exception
        print("  Attempting to get detailed error...")
        try:
            info = es.info()
        except Exception as info_error:
            print(f"  Detailed error: {type(info_error).__name__}: {info_error}")
            import traceback
            traceback.print_exc()
        
except Exception as e:
    print(f"  ❌ FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Using ElasticsearchVectorStore
print("\n[3] Testing ElasticsearchVectorStore...")
try:
    from retrieval.vector_store_es import ElasticsearchVectorStore
    
    store = ElasticsearchVectorStore()
    
    result = store.health_check()
    
    if result:
        print("  ✅ SUCCESS! VectorStore connection works")
        info = store.get_cluster_info()
        print(f"     Cluster: {info.get('cluster_name')}")
        print(f"     Version: {info.get('version')}")
    else:
        print("  ❌ FAILED: health_check() returned False")
        # Try to get more info
        print("  Attempting to get detailed error...")
        try:
            info = store.get_cluster_info()
            if info.get('status') == 'error':
                print(f"  Error from cluster_info: {info.get('error')}")
        except Exception as info_error:
            print(f"  Detailed error: {type(info_error).__name__}: {info_error}")
            import traceback
            traceback.print_exc()
        
except Exception as e:
    print(f"  ❌ FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Check if Elasticsearch is accessible via HTTP
print("\n[4] Testing HTTP accessibility...")
try:
    import urllib.request
    response = urllib.request.urlopen(ELASTICSEARCH_URL, timeout=5)
    print(f"  ✅ HTTP connection successful (Status: {response.getcode()})")
    content = response.read(200).decode('utf-8')
    print(f"  Response preview: {content[:100]}...")
except Exception as e:
    print(f"  ❌ HTTP connection failed: {type(e).__name__}: {e}")

print("\n" + "="*80)
print("TROUBLESHOOTING:")
print("="*80)
print("If ping() returns False but HTTP works, it might be:")
print("1. Elasticsearch 8.x requires different connection parameters")
print("2. Try using 'http://127.0.0.1:9200' instead of 'http://localhost:9200'")
print("3. Check if Elasticsearch container is fully started: docker logs elasticsearch")
print("="*80)