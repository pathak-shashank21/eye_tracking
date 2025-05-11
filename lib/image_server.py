import socket
import json

class ImageServer:
    def __init__(self, model_dir=None, port=11573):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", port))
        self.sock.settimeout(0.01)

    def get(self, frame=None):
        try:
            data, _ = self.sock.recvfrom(65536)
            parsed = json.loads(data.decode("utf-8"))

            if 'landmarks3d' in parsed and parsed['landmarks3d']:
                return [{
                    'landmarks': parsed['landmarks'],
                    'landmarks_3d': parsed['landmarks3d'],
                    'is_blinking': min(parsed.get("eye_l", 1.0), parsed.get("eye_r", 1.0)) < 0.2
                }]
            else:
                print("⚠️ Incomplete face data received")
                return []

        except socket.timeout:
            return []
        except Exception as e:
            print(f"⚠️ Error receiving UDP data: {e}")
            return []
