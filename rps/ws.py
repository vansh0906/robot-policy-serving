"""Minimal WebSocket (RFC 6455) server and client on asyncio streams, with no third-party dependencies.

Supports text and binary messages, fragmented messages, ping/pong and close. Client frames are masked, server frames
are not, as the RFC requires. Messages over MAX_MESSAGE bytes close the connection with code 1009.
"""
import asyncio
import base64
import hashlib
import os
import struct

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
MAX_MESSAGE = 1 << 20
OP_CONT, OP_TEXT, OP_BIN, OP_CLOSE, OP_PING, OP_PONG = 0x0, 0x1, 0x2, 0x8, 0x9, 0xA


class ConnectionClosed(Exception):
    pass


def accept_key(key):
    return base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()


class WebSocket:
    def __init__(self, reader, writer, is_client):
        self.reader, self.writer, self.is_client = reader, writer, is_client
        self.closed = False
        self._send_lock = asyncio.Lock()

    async def _read_frame(self):
        hdr = await self.reader.readexactly(2)
        fin, op = hdr[0] & 0x80, hdr[0] & 0x0F
        masked, n = hdr[1] & 0x80, hdr[1] & 0x7F
        if n == 126:
            n = struct.unpack("!H", await self.reader.readexactly(2))[0]
        elif n == 127:
            n = struct.unpack("!Q", await self.reader.readexactly(8))[0]
        if n > MAX_MESSAGE:
            await self.close(1009)
            raise ConnectionClosed("message too big")
        mask = await self.reader.readexactly(4) if masked else None
        data = await self.reader.readexactly(n)
        if mask:
            data = _unmask(data, mask)
        return bool(fin), op, data

    async def recv(self):
        """Next text (str) or binary (bytes) message; raises ConnectionClosed."""
        buf, first_op = bytearray(), None
        try:
            while True:
                fin, op, data = await self._read_frame()
                if op == OP_PING:
                    await self._send(OP_PONG, data)
                    continue
                if op == OP_PONG:
                    continue
                if op == OP_CLOSE:
                    if not self.closed:
                        await self.close(struct.unpack("!H", data[:2])[0] if len(data) >= 2 else 1000)
                    raise ConnectionClosed("closed by peer")
                if op != OP_CONT:
                    first_op = op
                buf += data
                if len(buf) > MAX_MESSAGE:
                    await self.close(1009)
                    raise ConnectionClosed("message too big")
                if fin:
                    return buf.decode() if first_op == OP_TEXT else bytes(buf)
        except (asyncio.IncompleteReadError, ConnectionError) as e:
            self.closed = True
            raise ConnectionClosed(str(e)) from None

    async def _send(self, op, payload):
        n = len(payload)
        head = bytes([0x80 | op])
        mbit = 0x80 if self.is_client else 0
        if n < 126:
            head += bytes([mbit | n])
        elif n < 65536:
            head += bytes([mbit | 126]) + struct.pack("!H", n)
        else:
            head += bytes([mbit | 127]) + struct.pack("!Q", n)
        if self.is_client:
            mask = os.urandom(4)
            head += mask
            payload = _unmask(payload, mask)
        async with self._send_lock:
            try:
                self.writer.write(head + payload)
                await self.writer.drain()
            except (ConnectionError, RuntimeError) as e:
                self.closed = True
                raise ConnectionClosed(str(e)) from None

    async def send(self, msg):
        if self.closed:
            raise ConnectionClosed("closed")
        if isinstance(msg, str):
            await self._send(OP_TEXT, msg.encode())
        else:
            await self._send(OP_BIN, bytes(msg))

    async def close(self, code=1000):
        if self.closed:
            return
        try:
            await self._send(OP_CLOSE, struct.pack("!H", code))
        except ConnectionClosed:
            pass
        self.closed = True
        try:
            self.writer.close()
        except Exception:
            pass


def _unmask(data, mask):
    """XOR with the 4-byte mask, done 8 bytes at a time through int.from_bytes for speed."""
    n = len(data)
    if n == 0:
        return b""
    m = (mask * ((n + 3) // 4))[:n]
    return (int.from_bytes(data, "little") ^ int.from_bytes(m, "little")).to_bytes(n, "little")


async def server_handshake(reader, writer, headers):
    key = headers.get("sec-websocket-key")
    if not key or headers.get("upgrade", "").lower() != "websocket":
        writer.write(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n")
        await writer.drain()
        return None
    writer.write(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                  f"Sec-WebSocket-Accept: {accept_key(key)}\r\n\r\n").encode())
    await writer.drain()
    return WebSocket(reader, writer, is_client=False)


async def connect(host, port, path):
    reader, writer = await asyncio.open_connection(host, port)
    sock = writer.get_extra_info("socket")
    if sock is not None:
        import socket
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    key = base64.b64encode(os.urandom(16)).decode()
    writer.write((f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                  f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
    await writer.drain()
    status = await reader.readline()
    headers = {}
    while True:
        line = await reader.readline()
        if line in (b"\r\n", b""):
            break
        k, _, v = line.decode().partition(":")
        headers[k.strip().lower()] = v.strip()
    if b" 101 " not in status or headers.get("sec-websocket-accept") != accept_key(key):
        writer.close()
        raise ConnectionError(f"handshake failed: {status!r}")
    return WebSocket(reader, writer, is_client=True)
