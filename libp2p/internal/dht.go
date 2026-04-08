package internal

import (
	"context"
	"fmt"
	"log"

	"github.com/ipfs/go-cid"
	dht "github.com/libp2p/go-libp2p-kad-dht"
	libp2phost "github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/multiformats/go-multiaddr"
	mh "github.com/multiformats/go-multihash"
)

// DHTManager wraps an IpfsDHT and its lifecycle context.
type DHTManager struct {
	kdht *dht.IpfsDHT
	ctx  context.Context
}

// NewDHTManager creates a Kademlia DHT attached to h.
// mode: "server" → ModeServer, "client" → ModeClient, anything else → ModeAutoServer.
// bootstrapAddrs is a list of multiaddr strings for initial bootstrap peers.
// Bootstrap connectivity failures are non-fatal and only logged.
func NewDHTManager(ctx context.Context, h libp2phost.Host, mode string, bootstrapAddrs []string) (*DHTManager, error) {
	var modeOpt dht.ModeOpt
	switch mode {
	case "server":
		modeOpt = dht.ModeServer
	case "client":
		modeOpt = dht.ModeClient
	default:
		modeOpt = dht.ModeAutoServer
	}

	kdht, err := dht.New(ctx, h, dht.Mode(modeOpt))
	if err != nil {
		return nil, fmt.Errorf("failed to create Kademlia DHT: %w", err)
	}

	// Connect to bootstrap peers (non-fatal on individual failure).
	for _, addrStr := range bootstrapAddrs {
		if addrStr == "" {
			continue
		}
		ma, err := multiaddr.NewMultiaddr(addrStr)
		if err != nil {
			log.Printf("DHTManager: skipping invalid bootstrap addr %q: %v", addrStr, err)
			continue
		}
		pi, err := peer.AddrInfoFromP2pAddr(ma)
		if err != nil {
			log.Printf("DHTManager: failed to extract peer info from %q: %v", addrStr, err)
			continue
		}
		if err := h.Connect(ctx, *pi); err != nil {
			log.Printf("DHTManager: could not connect to bootstrap peer %s: %v", pi.ID, err)
		} else {
			log.Printf("DHTManager: connected to bootstrap peer %s", pi.ID)
		}
	}

	if err := kdht.Bootstrap(ctx); err != nil {
		return nil, fmt.Errorf("DHT bootstrap failed: %w", err)
	}

	log.Printf("DHTManager: DHT started in mode=%s", mode)
	return &DHTManager{kdht: kdht, ctx: ctx}, nil
}

// contentCID converts an arbitrary string key into a CIDv1 (SHA2-256, raw codec).
func contentCID(key string) (cid.Cid, error) {
	hash, err := mh.Sum([]byte(key), mh.SHA2_256, -1)
	if err != nil {
		return cid.Undef, err
	}
	return cid.NewCidV1(cid.Raw, hash), nil
}

// Provide announces that this node provides the content identified by key.
func (m *DHTManager) Provide(key string) error {
	c, err := contentCID(key)
	if err != nil {
		return fmt.Errorf("contentCID: %w", err)
	}
	return m.kdht.Provide(m.ctx, c, true)
}

// FindProviders searches the DHT for peers that advertise the given key.
// At most limit results are returned.
func (m *DHTManager) FindProviders(key string, limit int) ([]peer.AddrInfo, error) {
	c, err := contentCID(key)
	if err != nil {
		return nil, fmt.Errorf("contentCID: %w", err)
	}
	ch := m.kdht.FindProvidersAsync(m.ctx, c, limit)
	var results []peer.AddrInfo
	for pi := range ch {
		results = append(results, pi)
	}
	return results, nil
}

// ConnectedPeers returns the number of peers in the DHT routing table.
func (m *DHTManager) ConnectedPeers() int {
	return m.kdht.RoutingTable().Size()
}

// Close shuts down the DHT.
func (m *DHTManager) Close() error {
	return m.kdht.Close()
}
