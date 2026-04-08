package internal

import (
	"encoding/binary"
	"encoding/json"
	"log"
	"net"
	"os"
	"sync"
	"sync/atomic"
)

// UDSMessage is the framed payload written to the Unix Domain Socket data plane.
type UDSMessage struct {
	MsgType     string `json:"msg_type"`      // "gossip" or "direct"
	TopicOrPeer string `json:"topic_or_peer"` // GossipSub topic or sender peer ID
	Data        string `json:"data"`          // JSON payload
	SenderID    string `json:"sender_id"`     // libp2p peer ID of sender
}

// UDSWriter manages a Unix Domain Socket listener that Python reads from via asyncio.
// Messages are framed as: [4-byte big-endian uint32 length][JSON payload].
type UDSWriter struct {
	path     string
	listener net.Listener
	conn     net.Conn
	mu       sync.Mutex
	dropped  int64
}

// NewUDSWriter removes any stale socket file, binds a new Unix listener, and
// returns a ready UDSWriter. Call AcceptConnection (in a goroutine) to block
// until Python connects.
func NewUDSWriter(path string) (*UDSWriter, error) {
	// Crash-recovery: remove stale socket file if present.
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return nil, err
	}

	ln, err := net.Listen("unix", path)
	if err != nil {
		return nil, err
	}

	return &UDSWriter{
		path:     path,
		listener: ln,
	}, nil
}

// AcceptConnection blocks until a client (Python) connects, then stores the
// connection for use by Write. Intended to be called in a goroutine.
func (w *UDSWriter) AcceptConnection() error {
	conn, err := w.listener.Accept()
	if err != nil {
		return err
	}

	w.mu.Lock()
	w.conn = conn
	w.mu.Unlock()

	log.Printf("UDSWriter: Python client connected on %s", w.path)
	return nil
}

// Write serialises msg to JSON, prepends a 4-byte big-endian length, and sends
// both over the Unix socket. If no client is connected yet, the message is
// silently dropped and the dropped counter is incremented.
func (w *UDSWriter) Write(msg UDSMessage) error {
	w.mu.Lock()
	conn := w.conn
	w.mu.Unlock()

	if conn == nil {
		atomic.AddInt64(&w.dropped, 1)
		return nil
	}

	payload, err := json.Marshal(msg)
	if err != nil {
		atomic.AddInt64(&w.dropped, 1)
		return err
	}

	// Build frame: 4-byte big-endian length prefix + JSON payload.
	frame := make([]byte, 4+len(payload))
	binary.BigEndian.PutUint32(frame[:4], uint32(len(payload)))
	copy(frame[4:], payload)

	if _, err := conn.Write(frame); err != nil {
		atomic.AddInt64(&w.dropped, 1)
		return err
	}

	return nil
}

// Dropped returns the total number of messages that could not be delivered.
func (w *UDSWriter) Dropped() int64 {
	return atomic.LoadInt64(&w.dropped)
}

// Close shuts down the connection, listener, and removes the socket file.
func (w *UDSWriter) Close() {
	w.mu.Lock()
	if w.conn != nil {
		w.conn.Close()
		w.conn = nil
	}
	w.mu.Unlock()

	if w.listener != nil {
		w.listener.Close()
	}

	os.Remove(w.path)
}
