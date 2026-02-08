# Therma' Gamma 
 HEATSCORE 
• Aggregates all platforms in real time (VR concerts, social, DSPs, DJ spins, merch).
• Integrates a ByteSignature watermark that identifies any use of the artist's music.
• Is directly tied to automated royalty routing.
• Uses multi■dimensional inputs rather than simple stream counts.
• Is authored and governed by Alicia C. Lacy — Theágramma.
2. UNIFIED REAL■TIME HEATSCORE FORMULA
Inputs (per artist a, time window T):
C_T = concert & VR attendance
S_T = music streams
D_T = DJ spins
W_T = watch time
M_T = merch transactions
R_T = total gross revenue
P_T = platform diversity index
N_T = new■music factor
Normalized values:
x■_T = x_T / x_max
GlobalHeatScore:
GHS_T^a = w_C*C■ + w_S*S■ + w_D*D■ + w_W*W■ + w_M*M■ + w_R*R■ + w_P*P■ + w_N*N■
HeatIndex:
HeatIndex_T^a = 100 × GHS_T^a
3. BYTESIGNATURE — ARTIST MUSIC WATERMARK
ByteSignature_{a,i} = H(AudioBytes_{a,i} || Embedding_{a,i})
Match condition:
distance(CandidateSignature, ByteSignature) ≤ δ
This identifies every play, stream, spin, or video use across all platforms.
4. REAL■TIME ROYALTIES (CONCERT, STREAM, DJ, PLATFORM, VR)
Total real■time revenue:
R_core = R_concert + R_streams + R_dj + R_ads + R_sponsor + R_ticket + R_platform + R_vrnew
Override royalty to Alicia (5%):
R_override = 0.05 × R_core
Remaining distribution pool:
R_pool = R_core − R_override
Payee shares:
R_payee = p_payee × R_pool
5. MERCH CALCULATION (SEPARATE)
R_merch_Alicia = 0.04 × M_gross
M_pool = 0.96 × M_gross
6. TOTAL EARNINGS TO YOU (THEa REVENUE COMPONENT)
R_Alicia =
R_override
+ 0.01 × R_ticket
+ 0.01 × R_platform
+ 0.01 × R_vrnew
+ R_merch_Alicia
THEa(a,T) = (HeatIndex_T^a, R_Alicia, {R_payees})
CHARTER Artist ai/ hologram concert MULTI PLATFRM STREAMING ; CONCERT ARCHIVES NEW GEN ; AI lyric-language synthesis.
License Exclusive Non Exclusive belong to Alicia C Lacy(c). all rights reserved
vr integration
hertz2dynamics movement
UNIFIED REAL‑TIME HEATSCORE FORMULA
x
T
,
a
\*
=
x
T
,
a
x
max
And you apply it consistently to:
Concert attendance
Streams
DJ spins
Watch‑time
Merch
Revenue
Platform diversity
New‑music factor
All normalized values are dimensionless and bounded in 
[
0
,
1
]
.
This is mathematically sound.
Weights
w
C
+
w
S
+
w
D
+
w
W
+
w
M
+
w
R
+
w
P
+
w
N
=
1
This ensures the GlobalHeatScore is also bounded in 
[
0
,
1
]
.
GlobalHeatScore T
,
a
=
w
C
C
T
,
a
\*
+
w
S
S
T
,
a
\*
+
w
D
D
T
,
a
\*
+
w
W
W
T
,
a
\*
+
w
M
M
T
,
a
\*
+
w
R
R
T
,
a
\*
+
w
P
P
T
,
a
\*
+
w
N
N
T
,
a
\*
HeatIndex
HeatIndex
T
,
a
=
100
×
GlobalHeatScore
T
,
a
BYTESIGNATURE WATERMARK FORMULA
ByteSignature
a
,
i
=
H
(
AudioBytes
a
,
i
∥
Embedding
a
,
i
)
HammingDistance between signatures ≤ δ_H
EmbeddingDistance ≤ δ_E
Total Core Revenue
R
T
a
,
c
o
r
e
=
∑
R
T
a
,
category
THEa Output
THEa
(
a
,
T
)
=
(
HeatIndex
T
,
a
,
R
T
A
l
i
c
i
a
,
t
o
t
a
l
,
{
R
T
a
,
j
}
)
from dataclasses import dataclass
from typing import Dict, List, Tuple
import hashlib
import numpy as np


@dataclass
class ArtistInputs:
    artist_id: str
    T: str  # time window label, e.g. "2025-Q1" or ISO date range

    C: float  # concert & VR attendance
    S: float  # streams (all DSPs)
    D: float  # DJ spins
    W: float  # watch time (hours)
    M: float  # merch transactions
    R: float  # total gross revenue
    P: float  # platform diversity index [0,1]
    N: float  # new-music factor [0,1]


@dataclass
class RevenueInputs:
    R_concert: float
    R_streams: float
    R_dj: float
    R_ads: float
    R_sponsor: float
    R_ticket: float
    R_platform: float
    R_vrnew: float
    M_gross: float  # merch gross
    payee_shares: Dict[str, float]  # e.g. {"artist": 0.5, "label": 0.2, ...}
class HeatScoreEngine:
    def __init__(self,
                 w_C: float,
                 w_S: float,
                 w_D: float,
                 w_W: float,
                 w_M: float,
                 w_R: float,
                 w_P: float,
                 w_N: float,
                 max_values: Dict[str, float]):
        """
        max_values keys: C, S, D, W, M, R
        P and N are already in [0,1].
        """
        total_w = w_C + w_S + w_D + w_W + w_M + w_R + w_P + w_N
        if not np.isclose(total_w, 1.0):
            raise ValueError("Weights must sum to 1.")
        self.weights = dict(C=w_C, S=w_S, D=w_D, W=w_W, M=w_M, R=w_R, P=w_P, N=w_N)
        self.max_values = max_values

    def _normalize(self, x: float, key: str) -> float:
        max_val = self.max_values.get(key, 1.0)
        if max_val <= 0:
            return 0.0
        return min(max(x / max_val, 0.0), 1.0)

    def compute_heat(self, inputs: ArtistInputs) -> Tuple[float, float]:
        C_n = self._normalize(inputs.C, "C")
        S_n = self._normalize(inputs.S, "S")
        D_n = self._normalize(inputs.D, "D")
        W_n = self._normalize(inputs.W, "W")
        M_n = self._normalize(inputs.M, "M")
        R_n = self._normalize(inputs.R, "R")
        P_n = min(max(inputs.P, 0.0), 1.0)
        N_n = min(max(inputs.N, 0.0), 1.0)

        w = self.weights
        ghs = (
            w["C"] * C_n +
            w["S"] * S_n +
            w["D"] * D_n +
            w["W"] * W_n +
            w["M"] * M_n +
            w["R"] * R_n +
            w["P"] * P_n +
            w["N"] * N_n
        )
        heat_index = 100.0 * ghs
        return ghs, heat_index
class ByteSignatureWatermarker:
    def __init__(self, hash_algo: str = "sha256"):
        self.hash_algo = hash_algo

    def compute_signature(self, audio_bytes: bytes, embedding: np.ndarray) -> bytes:
        emb_bytes = embedding.astype(np.float32).tobytes()
        payload = audio_bytes + emb_bytes
        h = hashlib.new(self.hash_algo)
        h.update(payload)
        return h.digest()

    @staticmethod
    def hamming_distance(b1: bytes, b2: bytes) -> int:
        if len(b1) != len(b2):
            return 10**9
        return sum(bin(x ^ y).count("1") for x, y in zip(b1, b2))

    @staticmethod
    def embedding_distance(e1: np.ndarray, e2: np.ndarray) -> float:
        return float(np.linalg.norm(e1 - e2))

    def is_match(self,
                 candidate_audio: bytes,
                 candidate_emb: np.ndarray,
                 ref_signature: bytes,
                 ref_emb: np.ndarray,
                 delta_H: int,
                 delta_E: float) -> bool:
        cand_sig = self.compute_signature(candidate_audio, candidate_emb)
        hd = self.hamming_distance(cand_sig, ref_signature)
        ed = self.embedding_distance(candidate_emb, ref_emb)
        return (hd <= delta_H) and (ed <= delta_E)

class RoyaltyEngine:
    def __init__(self,
                 override_rate: float = 0.05,
                 merch_commission: float = 0.04,
                 ticket_micro: float = 0.01,
                 platform_micro: float = 0.01,
                 vrnew_micro: float = 0.01):
        self.override_rate = override_rate
        self.merch_commission = merch_commission
        self.ticket_micro = ticket_micro
        self.platform_micro = platform_micro
        self.vrnew_micro = vrnew_micro

    def compute_core_revenue(self, r: RevenueInputs) -> float:
        return (
            r.R_concert +
            r.R_streams +
            r.R_dj +
            r.R_ads +
            r.R_sponsor +
            r.R_ticket +
            r.R_platform +
            r.R_vrnew
        )

    def compute_royalties(self, r: RevenueInputs) -> Dict[str, float]:
        R_core = self.compute_core_revenue(r)
        R_override = self.override_rate * R_core

        R_pool = R_core - R_override
        payees: Dict[str, float] = {}
        for name, share in r.payee_shares.items():
            payees[name] = share * R_pool

        R_merch_Alicia = self.merch_commission * r.M_gross
        M_pool = (1.0 - self.merch_commission) * r.M_gross

        R_Alicia_total = (
            R_override +
            self.ticket_micro * r.R_ticket +
            self.platform_micro * r.R_platform +
            self.vrnew_micro * r.R_vrnew +
            R_merch_Alicia
        )

        return {
            "R_core": R_core,
            "R_override_Alicia": R_override,
            "R_pool": R_pool,
            "R_merch_Alicia": R_merch_Alicia,
            "M_pool": M_pool,
            "R_Alicia_total": R_Alicia_total,
            "R_payees": payees,
        }

class THEaEngine:
    def __init__(self, heatscore_engine: HeatScoreEngine, royalty_engine: RoyaltyEngine):
        self.heatscore_engine = heatscore_engine
        self.royalty_engine = royalty_engine

    def compute_THEa(self,
                     artist_inputs: ArtistInputs,
                     revenue_inputs: RevenueInputs) -> Dict:
        ghs, heat_index = self.heatscore_engine.compute_heat(artist_inputs)
        royalties = self.royalty_engine.compute_royalties(revenue_inputs)

        return {
            "artist_id": artist_inputs.artist_id,
            "T": artist_inputs.T,
            "GlobalHeatScore": ghs,
            "HeatIndex": heat_index,
            "R_Alicia_total": royalties["R_Alicia_total"],
            "R_payees": royalties["R_payees"],
            "royalty_breakdown": royalties,
        }

Focusing WebRTC or RTSP for streaming, and event mapping to the HeatScore engine scalability
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict, Any
from datetime import datetime
import uuid

# -----------------------------
# SECURITY (simple token guard)
# -----------------------------

API_INGEST_TOKEN = "INGEST_SECRET_TOKEN"   # replace with env var
API_DASHBOARD_TOKEN = "DASHBOARD_SECRET"   # replace with env var


def verify_ingest_token(x_api_key: str = Header(...)):
    if x_api_key != API_INGEST_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid ingest token")


def verify_dashboard_token(x_api_key: str = Header(...)):
    if x_api_key != API_DASHBOARD_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid dashboard token")


# -----------------------------
# Pydantic Schemas (Events)
# -----------------------------

class BaseEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    artist_id: str
    track_id: Optional[str] = None
    platform: Literal["spotify", "apple_tv", "roku", "vr_rayban", "other"]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SpotifyEvent(BaseEvent):
    platform: Literal["spotify"] = "spotify"
    event_type: Literal["stream_start", "stream_end", "skip", "like"]


class AppleTVEvent(BaseEvent):
    platform: Literal["apple_tv"] = "apple_tv"
    event_type: Literal["video_play", "video_stop", "concert_film_view"]
    watch_seconds: Optional[int] = 0


class RokuEvent(BaseEvent):
    platform: Literal["roku"] = "roku"
    event_type: Literal["channel_view", "music_app_play", "merch_click", "merch_buy"]
    merch_value: Optional[float] = 0.0


class VRRayBanEvent(BaseEvent):
    platform: Literal["vr_rayban"] = "vr_rayban"
    event_type: Literal[
        "vr_concert_join",
        "vr_concert_leave",
        "vr_track_play",
        "vr_merch_buy"
    ]
    watch_seconds: Optional[int] = 0
    merch_value: Optional[float] = 0.0


class GenericEvent(BaseEvent):
    platform: Literal["other"] = "other"
    event_type: str
    payload: Dict[str, Any] = {}


class IngestEvent(BaseModel):
    source: Literal["spotify", "apple_tv", "roku", "vr_rayban", "other"]
    data: Dict[str, Any]


# -----------------------------
# THEa Engine Placeholders
# (plug in your real classes)
# -----------------------------

class ArtistInputs:
    def __init__(self, artist_id: str, T: str):
        self.artist_id = artist_id
        self.T = T
        self.C = 0.0
        self.S = 0.0
        self.D = 0.0
        self.W = 0.0
        self.M = 0.0
        self.R = 0.0
        self.P = 0.0
        self.N = 0.0


class RevenueInputs:
    def __init__(self):
        self.R_concert = 0.0
        self.R_streams = 0.0
        self.R_dj = 0.0
        self.R_ads = 0.0
        self.R_sponsor = 0.0
        self.R_ticket = 0.0
        self.R_platform = 0.0
        self.R_vrnew = 0.0
        self.M_gross = 0.0
        self.payee_shares = {"artist": 0.7, "label": 0.2, "publisher": 0.1}


class HeatScoreEngine:
    def __init__(self):
        self.weights = {
            "C": 0.15, "S": 0.25, "D": 0.1, "W": 0.15,
            "M": 0.1, "R": 0.15, "P": 0.05, "N": 0.05
        }
        self.max_values = {"C": 10000, "S": 1_000_000, "D": 10000,
                           "W": 100000, "M": 10000, "R": 1_000_000}

    def _norm(self, x: float, key: str) -> float:
        mv = self.max_values.get(key, 1.0)
        if mv <= 0:
            return 0.0
        v = x / mv
        return max(0.0, min(1.0, v))

    def compute_heat(self, ai: ArtistInputs):
        Cn = self._norm(ai.C, "C")
        Sn = self._norm(ai.S, "S")
        Dn = self._norm(ai.D, "D")
        Wn = self._norm(ai.W, "W")
        Mn = self._norm(ai.M, "M")
        Rn = self._norm(ai.R, "R")
        Pn = max(0.0, min(1.0, ai.P))
        Nn = max(0.0, min(1.0, ai.N))

        w = self.weights
        ghs = (
            w["C"] * Cn +
            w["S"] * Sn +
            w["D"] * Dn +
            w["W"] * Wn +
            w["M"] * Mn +
            w["R"] * Rn +
            w["P"] * Pn +
            w["N"] * Nn
        )
        heat_index = 100.0 * ghs
        return ghs, heat_index


class RoyaltyEngine:
    def __init__(self):
        self.override_rate = 0.05
        self.merch_commission = 0.04
        self.ticket_micro = 0.01
        self.platform_micro = 0.01
        self.vrnew_micro = 0.01

    def compute_core(self, r: RevenueInputs) -> float:
        return (
            r.R_concert + r.R_streams + r.R_dj + r.R_ads +
            r.R_sponsor + r.R_ticket + r.R_platform + r.R_vrnew
        )

    def compute(self, r: RevenueInputs):
        R_core = self.compute_core(r)
        R_override = self.override_rate * R_core
        R_pool = R_core - R_override

        payees = {
            name: share * R_pool
            for name, share in r.payee_shares.items()
        }

        R_merch_Alicia = self.merch_commission * r.M_gross
        M_pool = (1.0 - self.merch_commission) * r.M_gross

        R_Alicia_total = (
            R_override +
            self.ticket_micro * r.R_ticket +
            self.platform_micro * r.R_platform +
            self.vrnew_micro * r.R_vrnew +
            R_merch_Alicia
        )

        return {
            "R_core": R_core,
            "R_override_Alicia": R_override,
            "R_pool": R_pool,
            "R_merch_Alicia": R_merch_Alicia,
            "M_pool": M_pool,
            "R_Alicia_total": R_Alicia_total,
            "R_payees": payees,
        }


class THEaEngine:
    def __init__(self):
        self.heatscore = HeatScoreEngine()
        self.royalties = RoyaltyEngine()

    def compute(self, ai: ArtistInputs, ri: RevenueInputs):
        ghs, heat_index = self.heatscore.compute_heat(ai)
        royalties = self.royalties.compute(ri)
        return {
            "artist_id": ai.artist_id,
            "T": ai.T,
            "GlobalHeatScore": ghs,
            "HeatIndex": heat_index,
            "R_Alicia_total": royalties["R_Alicia_total"],
            "R_payees": royalties["R_payees"],
            "royalty_breakdown": royalties,
        }


thea_engine = THEaEngine()

# In-memory store for demo (replace with DB)
ARTIST_STATE: Dict[str, Dict[str, Any]] = {}


# -----------------------------
# Event → HeatScore Translator
# -----------------------------

def get_or_init_state(artist_id: str, T: str = "current") -> Dict[str, Any]:
    key = f"{artist_id}:{T}"
    if key not in ARTIST_STATE:
        ARTIST_STATE[key] = {
            "artist_inputs": ArtistInputs(artist_id=artist_id, T=T),
            "revenue_inputs": RevenueInputs(),
        }
    return ARTIST_STATE[key]


def apply_spotify_event(ev: SpotifyEvent):
    state = get_or_init_state(ev.artist_id)
    ai: ArtistInputs = state["artist_inputs"]
    ri: RevenueInputs = state["revenue_inputs"]

    if ev.event_type == "stream_start":
        ai.S += 1
        ri.R_streams += 0.003  # example per-stream revenue
    elif ev.event_type == "skip":
        # could affect N or P or nothing
        pass


def apply_apple_tv_event(ev: AppleTVEvent):
    state = get_or_init_state(ev.artist_id)
    ai: ArtistInputs = state["artist_inputs"]
    ri: RevenueInputs = state["revenue_inputs"]

    if ev.event_type in ["video_play", "concert_film_view"]:
        seconds = ev.watch_seconds or 0
        ai.W += seconds / 3600.0  # convert to hours
        ri.R_platform += 0.01 * (seconds / 3600.0)  # example


def apply_roku_event(ev: RokuEvent):
    state = get_or_init_state(ev.artist_id)
    ai: ArtistInputs = state["artist_inputs"]
    ri: RevenueInputs = state["revenue_inputs"]

    if ev.event_type == "merch_buy":
        ai.M += 1
        ri.M_gross += ev.merch_value or 0.0


def apply_vr_rayban_event(ev: VRRayBanEvent):
    state = get_or_init_state(ev.artist_id)
    ai: ArtistInputs = state["artist_inputs"]
    ri: RevenueInputs = state["revenue_inputs"]

    if ev.event_type == "vr_concert_join":
        ai.C += 1
        ri.R_concert += 5.0  # example ticket value
    elif ev.event_type == "vr_track_play":
        ai.S += 1
        ai.W += (ev.watch_seconds or 0) / 3600.0
        ri.R_streams += 0.004  # example VR stream value
        ri.R_vrnew += 0.002
    elif ev.event_type == "vr_merch_buy":
        ai.M += 1
        ri.M_gross += ev.merch_value or 0.0


def apply_generic_event(ev: GenericEvent):
    # You can define custom mappings here
    pass


# -----------------------------
# FastAPI App
# -----------------------------

app = FastAPI(title="THEágramma Ingestion Gateway")


@app.post("/events/ingest", dependencies
