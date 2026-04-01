"""PRSM ASCII banner — monochromatic geometric design.

The PRSM logo features a left-pointing chevron, a 3D rectangular form,
and a bottom accent line. All sharp angles, no curves. Isometric feel.
Reference: docs/assets/PRSM_Logo_Dark.png / PRSM_Logo_Light.png
"""

# ──────────────────────────────────────────────────────────────────────
# Premium geometric isometric banner — inspired by the actual PRSM logo
# Left chevron + 3D block + clean wordmark with geometric spacing
# ──────────────────────────────────────────────────────────────────────

PRSM_BANNER = r"""
            ╱╲
           ╱  ╲         ┌────────────┐
          ╱    ╲        │            │
         ╱      ╲───────┤            │
        ╱       ╱       │            │
       ╱       ╱        └────────────┘
      ╱       ╱
     ╱       ╱
    ╱───────╱
   ▔▔▔▔▔▔▔▔▔

    ╔═╗ ╔═╗ ╔═╗ ╔╗╔╗
    ╠═╝ ╠╦╝ ╚═╗ ║╚╝║
    ╩   ╩╚═ ╚═╝ ╩  ╩
"""

# Geometric icon mark only (no wordmark)
PRSM_ICON = r"""
            ╱╲
           ╱  ╲         ┌────────────┐
          ╱    ╲        │            │
         ╱      ╲───────┤            │
        ╱       ╱       │            │
       ╱       ╱        └────────────┘
      ╱       ╱
     ╱       ╱
    ╱───────╱
   ▔▔▔▔▔▔▔▔▔
"""

# Compact version for narrow terminals (<60 cols)
PRSM_BANNER_COMPACT = r"""
     ╱╲    ┌──┐
    ╱  ╲───┤  │
   ╱   ╱   └──┘
  ╱───╱

  P R S M
"""

# Minimal one-liner
PRSM_INLINE = "◇ PRSM"

TAGLINES = [
    "Decentralized AI infrastructure.",
    "Your compute. Your data. Your network.",
    "Intelligence, distributed.",
    "The open AI network.",
    "Compute without borders.",
]

# Thin geometric separator
RULE = "─" * 48
