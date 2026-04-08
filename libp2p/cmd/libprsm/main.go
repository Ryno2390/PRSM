package main

/*
#include <stdlib.h>
*/
import "C"
import (
	"log"
	"unsafe"

	"github.com/Ryno2390/PRSM/libp2p/internal"
)

// PrsmStart creates a placeholder Host, registers it, and returns a handle.
// Parameters are reserved for future real libp2p integration (Task 2).
//
//export PrsmStart
func PrsmStart(ed25519Key *C.char, listenPort C.int, bootstrapAddrs *C.char, udsPath *C.char) C.int {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("PrsmStart: recovered from panic: %v", r)
		}
	}()

	host := &internal.Host{
		ListenPort: int(listenPort),
		PeerID:     "", // populated in Task 2 when real libp2p host is started
	}

	handle := internal.Register(host)
	return C.int(handle)
}

// PrsmStop removes the host associated with handle from the registry.
// Returns 0 on success, -1 if the handle is not found.
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
	internal.Remove(int(handle))
	return C.int(0)
}

// PrsmPeerCount returns the number of connected peers for the given handle.
// Returns 0 as a placeholder until real libp2p integration is complete (Task 2).
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
	// Placeholder — real peer count will be returned in Task 2.
	return C.int(0)
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
