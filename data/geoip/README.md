# GeoIP databases

This directory holds operator-downloaded GeoIP databases used by
`prsm.node.jurisdiction_filter.MaxMindGeoIPResolver` for peer-location
lookups.

**The databases themselves are not in this repository.** MaxMind's
GeoLite2 End-User License Agreement prohibits redistribution, so each
operator downloads their own copy. The `.mmdb`, `.tar.gz`, `.md5`, and
`.sha256` files are gitignored (see root `.gitignore`).

## Why this matters

The R9 peer-jurisdiction filter is **optional** and **operator-local** —
only operators who have decided (based on their own compliance analysis
+ legal advice) to exclude or restrict peers by country need this file.
Operators who don't configure a jurisdiction policy don't need it at
all. See
[`docs/2026-04-23-r9-transport-censorship-resistance-scoping.md`](../../docs/2026-04-23-r9-transport-censorship-resistance-scoping.md)
and
[`docs/2026-04-23-prsm-policy-jurisdiction-1.md`](../../docs/2026-04-23-prsm-policy-jurisdiction-1.md)
for the full design + Foundation boundary policy.

## Setup

### 1. MaxMind account

Sign up at https://www.maxmind.com/en/geolite2/signup (free tier).
Accept the GeoLite2 EULA when prompted.

### 2. Download the Country database

Either:

- **Manual (easiest):** log into
  https://www.maxmind.com/en/accounts/current/geoip/downloads and click
  **Download GZIP** on the **GeoLite2 Country** row.

- **Automated (recommended for production):** generate a license key in
  the account dashboard and use `geoipupdate`
  (https://github.com/maxmind/geoipupdate) as a cron job / systemd
  timer.

### 3. Extract into this directory

```bash
tar -xzf /path/to/GeoLite2-Country_YYYYMMDD.tar.gz --strip-components=1 \
    -C data/geoip/ GeoLite2-Country_YYYYMMDD/GeoLite2-Country.mmdb
```

Final layout:

```
data/geoip/
├── README.md
└── GeoLite2-Country.mmdb  (≈ 9 MB, gitignored)
```

### 4. Install the Python reader

```bash
pip install maxminddb
```

(Already transitively pulled by `geoip2>=4.7.0` in `requirements.txt`.)

### 5. Verify

```bash
python - <<'PY'
import maxminddb
with maxminddb.open_database("data/geoip/GeoLite2-Country.mmdb") as r:
    rec = r.get("8.8.8.8")
    print(rec["country"]["iso_code"])  # expected: US
PY
```

## Wiring into the jurisdiction filter

```python
from pathlib import Path
from prsm.node.jurisdiction_filter import (
    MaxMindGeoIPResolver,
    StaticGeoIPResolver,
    ChainedGeoIPResolver,
    PeerJurisdictionFilter,
)

resolver = ChainedGeoIPResolver([
    StaticGeoIPResolver({"192.0.2.1": "US"}),  # operator overrides first
    MaxMindGeoIPResolver(Path("data/geoip/GeoLite2-Country.mmdb")),
])

jurisdiction_filter = PeerJurisdictionFilter(
    resolver=resolver,
    excluded={"XX", "YY"},  # operator-configured; empty by default
    strict=False,
)
```

## Update cadence

MaxMind releases fresh data **twice a week** (Tuesdays and Fridays).
For defensive peer filtering, weekly updates are sufficient. Operators
running compliance-critical filters should automate via `geoipupdate`.

## Database variant options

The filter only needs **country-level** data, so `GeoLite2-Country`
(~9 MB) is the right file. Do **not** download `GeoLite2-City`
(~80 MB) unless you have a separate use case — the extra city-level
precision is not used by PRSM's jurisdiction filter and incurs larger
RAM + disk footprint.

## License

GeoLite2 databases are distributed by MaxMind under the GeoLite2
End-User License Agreement (EULA). PRSM does not redistribute these
files. Each operator is responsible for complying with the MaxMind
EULA, including attribution requirements when displaying derived
geolocation information publicly.

Attribution string (if your operator dashboard surfaces country data
to users):

> This product includes GeoLite2 data created by MaxMind, available
> from https://www.maxmind.com.
