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
