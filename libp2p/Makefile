.PHONY: build clean

GOOS ?= $(shell go env GOOS)
GOARCH ?= $(shell go env GOARCH)

ifeq ($(GOOS),darwin)
  EXT = .dylib
else ifeq ($(GOOS),windows)
  EXT = .dll
else
  EXT = .so
endif

OUTPUT = build/libprsm_p2p_$(GOOS)_$(GOARCH)$(EXT)

build:
	CGO_ENABLED=1 GOOS=$(GOOS) GOARCH=$(GOARCH) \
		go build -buildmode=c-shared -o $(OUTPUT) ./cmd/libprsm/
	@echo "Built $(OUTPUT)"

clean:
	rm -rf build/
