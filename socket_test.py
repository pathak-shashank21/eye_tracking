import socket

UDP_IP = "127.0.0.1"
UDP_PORT = 11574

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(2)  # Timeout to avoid indefinite blocking

while True:
    try:
        data, addr = sock.recvfrom(65536)  # Buffer size
        print(f"Received data: {data}")
    except socket.timeout:
        print("No data received within timeout period.")
