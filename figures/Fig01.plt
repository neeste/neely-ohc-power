; fig1.plt - JASA Fig. 1 in PLT
; Structure of the model family: signal path with OHC feedback.
; the section box (the subject of the paper) carries a heavier outline;
; main-path boxes blue, OHC feedback box red.
; Lettering caption-size at double-column width (6.69 in); bold titles msglwt 1.3.
;
head=0 : clip=0
xllc=0.15 : yllc=0.45 : xlen=9.57 : ylen=3.55
xmin=0 : xmax=9.57 : ymin=0 : ymax=3.55
xaxis=0 : yaxis=0 : taxis=0 : raxis=0

; main signal-path boxes (blue outlines)
pltcol=1 : pltlwt=1.5 : pltype=rect
0.100 1.30
1.550 2.55
plot
1.850 1.30
3.750 2.55
plot
6.350 1.30
7.950 2.55
plot
8.250 1.30
9.450 2.55
plot
; 3-DOF section box, heavier outline
pltlwt=3
4.050 1.30
6.050 2.55
plot

; OHC feedback box (red)
pltcol=4 : pltlwt=1.5
4.030 0.08
6.070 1.05
plot

; signal-path arrows (black)
pltype=lines : aronum=1 : arosiz=1.2 : pltcol=0 : pltlwt=1.5
1.550 1.925
1.850 1.925
plot
3.750 1.925
4.050 1.925
plot
6.050 1.925
6.350 1.925
plot
7.950 1.925
8.250 1.925
plot

; observable arrows (blue, up)
pltcol=1
5.050 2.55
5.050 3.10
plot
8.850 2.55
8.850 3.10
plot

; feedback arrows (red, up)
pltcol=4
4.650 1.05
4.650 1.30
plot
5.450 1.05
5.450 1.30
plot
aronum=0

; text
msgdat=yes : mhal=cen : mval=half
msgcol=0 : msgsiz=-0.144 : msglwt=1.3
0.825 2.250 "Stapes"
2.800 2.250 "Fluid line"
5.050 2.250 "Section"
7.150 2.250 "Read-out"
8.850 2.250 "Neural"
msgcol=8 : msgsiz=-0.144 : msglwt=1
0.825 1.955 "pressure"
0.825 1.655 "source"
2.800 1.955 "1-D,"
2.800 1.655 "long-wave"
5.050 1.955 "BM, RL, TM"
5.050 1.655 "shear divider"
7.150 1.955 "bundle"
7.150 1.655 "shear"
8.850 1.805 "output"
msgcol=0 : msgsiz=-0.144 : msglwt=1.3
5.050 0.78 "OHC force"
msgcol=8 : msgsiz=-0.144 : msglwt=1
5.050 0.50 "on BM and RL,"
5.050 0.26 "saturating"
msgcol=1 : msgsiz=-0.144 : msglwt=1 : mhal=cen
5.050 3.270 "BM and RL motion"
mhal=rig
9.500 3.270 "thresholds, wave V"
