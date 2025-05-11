# test_udp_listener.py
import socket
import json

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('127.0.0.1', 11573))

while True:
    data, _ = sock.recvfrom(65536)
    print("✅ Got packet:", json.loads(data.decode('utf-8')))
