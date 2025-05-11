import socket
import json
import time

UDP_IP = "127.0.0.1"  # Make sure this matches the IP OpenSeeFace is sending to
UDP_PORT = 11573

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(5)  # Increased timeout to give OpenSeeFace more time to send data

def receive_data():
    while True:
        try:
            data, addr = sock.recvfrom(65536)  # Buffer size
            print(f"Received data from {addr}")
            
            # Debug: print raw data
            print("Raw Data:", data)
            
            face_data = json.loads(data.decode("utf-8"))
            print("Decoded JSON Data:", face_data)
            
            if isinstance(face_data, dict) and "landmarks3d" in face_data:
                print("Valid data received with 3D landmarks:")
                print(face_data["landmarks3d"])
            else:
                print("❌ Invalid data structure or no landmarks3d found")
        except socket.timeout:
            print("❌ Timeout: No data received within the timeout period.")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("Listening for data...")
    receive_data()
