package main

/*
#include <stdlib.h>
*/
import "C"
import (
	"context"
	"log"
	"unsafe"

	"github.com/Ryno2390/PRSM/libp2p/internal"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/multiformats/go-multiaddr"
)

// PrsmStart creates a libp2p host with the given Ed25519 key and listen port,
// registers it, and returns a handle. ed25519Key must point to 64 raw bytes.
//
//export PrsmStart
func PrsmStart(ed25519Key *C.char, listenPort C.int, bootstrapAddrs *C.char, udsPath *C.char) C.int {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("PrsmStart: recovered from panic: %v", r)
		}
	}()

	keyBytes := C.GoBytes(unsafe.Pointer(ed25519Key), 64)

	p2pHost, err := internal.CreateHost(keyBytes, int(listenPort))
	if err != nil {
		log.Printf("PrsmStart: failed to create host: %v", err)
		return C.int(0)
	}

	ctx, cancel := context.WithCancel(context.Background())

	host := &internal.Host{
		ListenPort: int(listenPort),
		PeerID:     p2pHost.ID().String(),
		Ctx:        ctx,
		Cancel:     cancel,
		P2PHost:    p2pHost,
	}

	handle := internal.Register(host)
	return C.int(handle)
}

// PrsmStop shuts down the libp2p host associated with handle and removes it
// from the registry. Returns 0 on success, -1 if the handle is not found.
//
//export PrsmStop
func PrsmStop(handle C.int) C.int {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("PrsmStop: recovered from panic: %v", r)
		}
	}()

	h := internal.Get(int(handle))
	if h == nil {
		return C.int(-1)
	}

	h.Cancel()
	if err := h.P2PHost.Close(); err != nil {
		log.Printf("PrsmStop: error closing host: %v", err)
	}
	internal.Remove(int(handle))
	return C.int(0)
}

// PrsmPeerCount returns the number of connected peers for the given handle.
// Returns -1 if the handle is not found.
//
//export PrsmPeerCount
func PrsmPeerCount(handle C.int) C.int {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("PrsmPeerCount: recovered from panic: %v", r)
		}
	}()

	h := internal.Get(int(handle))
	if h == nil {
		return C.int(-1)
	}
	return C.int(len(h.P2PHost.Network().Peers()))
}

// PrsmConnect dials the peer at the given multiaddr string and returns the
// peer's ID string on success, or nil on failure. The returned string must be
// freed with PrsmFree.
//
//export PrsmConnect
func PrsmConnect(handle C.int, maddr *C.char) *C.char {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("PrsmConnect: recovered from panic: %v", r)
		}
	}()

	h := internal.Get(int(handle))
	if h == nil {
		log.Printf("PrsmConnect: handle %d not found", int(handle))
		return nil
	}

	maddrStr := C.GoString(maddr)
	ma, err := multiaddr.NewMultiaddr(maddrStr)
	if err != nil {
		log.Printf("PrsmConnect: invalid multiaddr %q: %v", maddrStr, err)
		return nil
	}

	peerInfo, err := peer.AddrInfoFromP2pAddr(ma)
	if err != nil {
		log.Printf("PrsmConnect: failed to extract peer info from %q: %v", maddrStr, err)
		return nil
	}

	if err := h.P2PHost.Connect(h.Ctx, *peerInfo); err != nil {
		log.Printf("PrsmConnect: failed to connect to %s: %v", peerInfo.ID, err)
		return nil
	}

	return C.CString(peerInfo.ID.String())
}

// PrsmFree releases memory allocated by C.CString.
//
//export PrsmFree
func PrsmFree(ptr *C.char) {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("PrsmFree: recovered from panic: %v", r)
		}
	}()

	C.free(unsafe.Pointer(ptr))
}

// main is required for c-shared build mode.
func main() {}
