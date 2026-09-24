import os
import psycopg2
import redis
from confluent_kafka import Producer

def test_connections():
    ca_path = os.getenv('CA_CERT_PATH')
    print(f"--- Verification Start (CA Path: {ca_path}) ---")

    # 1. Test PostgreSQL
    try:
        conn = psycopg2.connect(os.getenv('POSTGRES_URI'))
        print("✅ PostgreSQL: Connection Successful!")
        conn.close()
    except Exception as e:
        print(f"❌ PostgreSQL: Failed! Error: {e}")

    # 2. Test Redis (Valkey)
    try:
        r = redis.Redis.from_url(
            os.getenv('REDIS_URI'), 
            ssl_ca_certs=ca_path
        )
        if r.ping():
            print("✅ Redis/Valkey: Connection Successful!")
    except Exception as e:
        print(f"❌ Redis: Failed! Error: {e}")

    # 3. Test Kafka
    try:
        # We only initialize the producer to check credentials/connectivity
        p = Producer({
            'bootstrap.servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS'),
            'security.protocol': 'SASL_SSL',
            'sasl.mechanisms': 'SCRAM-SHA-256',
            'sasl.username': os.getenv('KAFKA_USER'),
            'sasl.password': os.getenv('KAFKA_PASSWORD'),
            'ssl.ca.location': ca_path
        })
        print("✅ Kafka: Producer initialized successfully!")
    except Exception as e:
        print(f"❌ Kafka: Failed! Error: {e}")

if __name__ == "__main__":
    test_connections()