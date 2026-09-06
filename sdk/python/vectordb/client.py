import socket
import struct
from typing import List, Tuple, Optional

class VectorDBClient:
    """Python Client for the C++ High-Throughput Distributed Vector Search Engine."""

    CMD_INSERT = 0x01
    CMD_SEARCH = 0x02
    CMD_PING   = 0x03
    CMD_PONG   = 0x04
    CMD_STATS  = 0x05

    def __init__(self, host: str = "127.0.0.1", port: int = 9000):
        self.host = host
        self.port = port
        self.sock: Optional[socket.socket] = None

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None

    def ping(self) -> bool:
        if not self.sock:
            self.connect()
        header = struct.pack("!BI", self.CMD_PING, 0)
        self.sock.sendall(header)
        res = self.sock.recv(1)
        return len(res) == 1 and res[0] == self.CMD_PONG

    def stats(self) -> int:
        if not self.sock:
            self.connect()
        header = struct.pack("!BI", self.CMD_STATS, 0)
        self.sock.sendall(header)
        res = self.sock.recv(8)
        return struct.unpack("<Q", res)[0]

    def insert(self, vector_id: int, vector: List[float]) -> bool:
        if not self.sock:
            self.connect()
        dim = len(vector)
        payload = struct.pack("<QI", vector_id, dim) + struct.pack(f"<{dim}f", *vector)
        header = struct.pack("!BI", self.CMD_INSERT, len(payload))
        self.sock.sendall(header + payload)
        res = self.sock.recv(1)
        return len(res) == 1 and res[0] == 0x00

    def search(self, query_vector: List[float], k: int = 5) -> List[Tuple[int, float]]:
        if not self.sock:
            self.connect()
        dim = len(query_vector)
        payload = struct.pack("<II", k, dim) + struct.pack(f"<{dim}f", *query_vector)
        header = struct.pack("!BI", self.CMD_SEARCH, len(payload))
        self.sock.sendall(header + payload)

        num_res_data = self.sock.recv(4)
        if len(num_res_data) < 4:
            return []
        num_res = struct.unpack("<I", num_res_data)[0]

        results = []
        for _ in range(num_res):
            rec = self.sock.recv(12) # uint64 id (8B) + float distance (4B)
            if len(rec) < 12:
                break
            vec_id, dist = struct.unpack("<Qf", rec)
            results.append((vec_id, dist))
        return results

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
