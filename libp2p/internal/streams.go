package internal

import (
	"bufio"
	"context"
	"fmt"
	"log"

	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/protocol"
)

const directProtocol = protocol.ID("/prsm/direct/1.0.0")

// StreamManager handles direct peer-to-peer message streams using a custom protocol.
type StreamManager struct {
	host host.Host
	uds  *UDSWriter
	ctx  context.Context
}

// NewStreamManager creates a StreamManager and registers the direct protocol handler on the host.
func NewStreamManager(ctx context.Context, h host.Host, uds *UDSWriter) *StreamManager {
	sm := &StreamManager{
		host: h,
		uds:  uds,
		ctx:  ctx,
	}
	h.SetStreamHandler(directProtocol, sm.handleStream)
	return sm
}

// handleStream reads newline-delimited messages from an incoming stream and
// routes each one to the UDS writer with msg_type="direct".
func (sm *StreamManager) handleStream(s network.Stream) {
	defer s.Close()

	senderID := s.Conn().RemotePeer().String()
	scanner := bufio.NewScanner(s)

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}

		if sm.uds == nil {
			continue
		}

		msg := UDSMessage{
			MsgType:     "direct",
			TopicOrPeer: senderID,
			Data:        line,
			SenderID:    senderID,
		}
		if err := sm.uds.Write(msg); err != nil {
			log.Printf("StreamManager: failed to write direct msg to UDS: %v", err)
		}
	}

	if err := scanner.Err(); err != nil {
		log.Printf("StreamManager: stream read error from %s: %v", senderID, err)
	}
}

// Send opens a new stream to peerID, writes data followed by a newline, then closes the stream.
func (sm *StreamManager) Send(peerID peer.ID, data []byte) error {
	s, err := sm.host.NewStream(sm.ctx, peerID, directProtocol)
	if err != nil {
		return fmt.Errorf("failed to open stream to %s: %w", peerID, err)
	}
	defer s.Close()

	// Write data + newline delimiter.
	buf := make([]byte, len(data)+1)
	copy(buf, data)
	buf[len(data)] = '\n'

	if _, err := s.Write(buf); err != nil {
		return fmt.Errorf("failed to write to stream: %w", err)
	}

	return nil
}
