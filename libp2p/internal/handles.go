package internal

import (
	"context"
	"sync"
	"sync/atomic"

	libp2phost "github.com/libp2p/go-libp2p/core/host"
)

// Host wraps a libp2p host with its lifecycle context and registry metadata.
type Host struct {
	ListenPort int
	PeerID     string
	Ctx        context.Context
	Cancel     context.CancelFunc
	P2PHost    libp2phost.Host
}

var (
	registry  sync.Map
	handleGen atomic.Int32
)

// Register stores a Host in the registry and returns its handle.
// Handle values start at 1; handle 0 is reserved as invalid.
func Register(h *Host) int {
	handle := int(handleGen.Add(1))
	registry.Store(handle, h)
	return handle
}

// Get retrieves a Host by handle. Returns nil if not found.
func Get(handle int) *Host {
	val, ok := registry.Load(handle)
	if !ok {
		return nil
	}
	host, _ := val.(*Host)
	return host
}

// Remove deletes a Host from the registry.
func Remove(handle int) {
	registry.Delete(handle)
}
