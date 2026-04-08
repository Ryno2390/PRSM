package internal

import (
	"fmt"
	"log"

	libp2p "github.com/libp2p/go-libp2p"
	libp2pcrypto "github.com/libp2p/go-libp2p/core/crypto"
	libp2phost "github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/p2p/muxer/yamux"
	"github.com/libp2p/go-libp2p/p2p/security/noise"
)

// CreateHost creates a libp2p host from a raw 64-byte Ed25519 private key.
// The key must be in Go's standard format: 32-byte seed + 32-byte public key.
// It listens on both QUIC (primary) and TCP (fallback) transports.
// Additional libp2p options (e.g. NAT traversal) can be passed via extraOpts.
func CreateHost(ed25519Raw []byte, listenPort int, extraOpts ...libp2p.Option) (libp2phost.Host, error) {
	if len(ed25519Raw) != 64 {
		return nil, fmt.Errorf("ed25519 key must be exactly 64 bytes, got %d", len(ed25519Raw))
	}

	privKey, err := libp2pcrypto.UnmarshalEd25519PrivateKey(ed25519Raw)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal Ed25519 private key: %w", err)
	}

	quicAddr := fmt.Sprintf("/ip4/0.0.0.0/udp/%d/quic-v1", listenPort)
	tcpAddr := fmt.Sprintf("/ip4/0.0.0.0/tcp/%d", listenPort)

	baseOpts := []libp2p.Option{
		libp2p.Identity(privKey),
		libp2p.ListenAddrStrings(quicAddr, tcpAddr),
		libp2p.Security(noise.ID, noise.New),
		libp2p.Muxer(yamux.ID, yamux.DefaultTransport),
		libp2p.DisableMetrics(),
	}
	baseOpts = append(baseOpts, extraOpts...)

	h, err := libp2p.New(baseOpts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create libp2p host: %w", err)
	}

	log.Printf("libp2p host started: peerID=%s", h.ID().String())
	for _, addr := range h.Addrs() {
		log.Printf("  listening on %s", addr)
	}

	return h, nil
}
