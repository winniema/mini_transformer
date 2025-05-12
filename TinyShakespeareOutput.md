# Sample of Outputs of MiniTransformer on TinyShakespeare Input

## Single Head of Attention

step 0: train loss 4.1719, val loss 4.1712
step 300: train loss 2.9566, val loss 2.9717
step 600: train loss 2.6935, val loss 2.7057
step 900: train loss 2.5970, val loss 2.6022
step 1200: train loss 2.5393, val loss 2.5439
step 1500: train loss 2.5093, val loss 2.5173
step 1800: train loss 2.4893, val loss 2.4940
step 2100: train loss 2.4704, val loss 2.4830
step 2400: train loss 2.4589, val loss 2.4519
step 2700: train loss 2.4511, val loss 2.4594

MI me isevo ant y yurtor to itsh, thien; beawe ocere C:
ARODaland wos fom tairans isoe.

S!
ngt tas I ell un: ls lrout st ow, owiwem,
hr.

BELLI:
BES allis, ber wan munenc, ban thes cem trave sichinak yue alen be thengoker ee outhe to thantr bu.

He aeavo; hat fore be ul shord whito t't pono hee erl yes ise junndlat brise ane in in aprlifult, tin bend, ed ir's yo asinod yoowu ous atho thakinwifet ns, Lhe f ath mullale rand qusos wall that pirat.

IUEASeyerl
RENGIHen hith aringee freal I ILO:
Oif

## Multi-headed Attention (num heads = 4)
- Attention blocks are concatenated, but with no linear projection at the end

step 0: train loss 4.2239, val loss 4.2239
step 500: train loss 2.6650, val loss 2.6660
step 1000: train loss 2.5041, val loss 2.5073
step 1500: train loss 2.4292, val loss 2.4355
step 2000: train loss 2.3766, val loss 2.4019
step 2500: train loss 2.3552, val loss 2.3579
step 3000: train loss 2.3208, val loss 2.3388
step 3500: train loss 2.3000, val loss 2.3163
step 4000: train loss 2.2691, val loss 2.2980
step 4500: train loss 2.2613, val loss 2.2859

Whys an Hit hadur to Bearstoo thour ha tores;
Hores and acich nooon for,
An this;
What wonsemy wo othy a So this fe loss; jay to so anweer, Shee himeesse
Twell,
Dame,
nriver hully, thownes whawlellidl.

DI:

T:
An is meas la lexokt pomy fo-wils myou: the, shallawk oro to I'I thaccum; I ber pit of cakak, to kathins:
So tinced tiow, a thoushus ther topire ther hak dalleag, hy fect?
Wous werse go man.

BLE IONGWHAOorw heawin not,
wit the thavoll se,
Malld; and
Wis not base Aliof meadce the morn tho


## Adding Feed Forward Layer (projecting into 4x bigger dim than n_embd)

step 0: train loss 4.1511, val loss 4.1511
step 500: train loss 2.5427, val loss 2.5454
step 1000: train loss 2.4042, val loss 2.4082
step 1500: train loss 2.3276, val loss 2.3347
step 2000: train loss 2.2785, val loss 2.3050
step 2500: train loss 2.2514, val loss 2.2645
step 3000: train loss 2.2213, val loss 2.2560
step 3500: train loss 2.2080, val loss 2.2321
step 4000: train loss 2.1750, val loss 2.2091
step 4500: train loss 2.1659, val loss 2.1898
step 5000: train loss 2.1489, val loss 2.1859

Lon RILONENEd, to troke be.

SIENT:
Gond.

KITBAUS:
Med
Rou the nelw che pom to up deeasts and I him, wip;
Golautent:
Aod not knight to bus,
Orce?

JOSALAME:
Wace shims;
Bugh noche wants, oldy ummor! Fnotk
Roth awul Geare Mlover ast,
Her, preire, com muck: it your but eor triwnour comepnother a hor at boved is of wel;
Ah'd I him, andiod ble my mond a, Brrep.

CRIG oof save, O, Cout'd;
Go ESts saing beimfen, beare, thech dose; ood shan :
So pie reir, Why,-

OPLARIXZALABAREH:


## Two Blocks of Multi-attention and Feed Forward
- Two repeated blocks of multi-headed attention and feed forward layers with residual pathways

step 0: train loss 4.6168, val loss 4.6233
step 500: train loss 2.4464, val loss 2.4490
step 1000: train loss 2.2817, val loss 2.3051
step 1500: train loss 2.2294, val loss 2.2527
step 2000: train loss 2.1857, val loss 2.2148
step 2500: train loss 2.1464, val loss 2.1909
step 3000: train loss 2.1123, val loss 2.1830
step 3500: train loss 2.0800, val loss 2.1502
step 4000: train loss 2.0707, val loss 2.1427
step 4500: train loss 2.0539, val loss 2.1365
step 5000: train loss 2.0335, val loss 2.1075


GSICLIFFORTINGA:
A BROLOUTHAMICK:
And my lang to meast saitius forso en,
The brise, have teopess a fat wauull:
The shilk to kan tis yof owsent
Ber't;
And my us ssuckeates that by tewelan's I wabs gre we my perfome anowith have treove tele ountion's trong, bawner,
Would lat tre wast?
An her trou hough hall tithis baun and miscare lepglant.

DUCEO:
Wet,
And do howne then theath.
God
Is Puch
Theif comediver ye.
And I reasen at, bed tre's ibrets to searr
Wet well kfals.


## Adding layer norm

step 0: train loss 4.2892, val loss 4.2915
step 250: train loss 2.6314, val loss 2.6468
step 500: train loss 2.4567, val loss 2.4492
step 750: train loss 2.3660, val loss 2.3621
step 1000: train loss 2.2855, val loss 2.3040
step 1250: train loss 2.2601, val loss 2.2756
step 1500: train loss 2.2087, val loss 2.2362
step 1750: train loss 2.1687, val loss 2.2095
step 2000: train loss 2.1692, val loss 2.2000
step 2250: train loss 2.1542, val loss 2.1817
step 2500: train loss 2.1313, val loss 2.1702
step 2750: train loss 2.1135, val loss 2.1549
step 3000: train loss 2.0991, val loss 2.1539
step 3250: train loss 2.0805, val loss 2.1294
step 3500: train loss 2.0704, val loss 2.1300
step 3750: train loss 2.0536, val loss 2.1120
step 4000: train loss 2.0688, val loss 2.1186
step 4250: train loss 2.0374, val loss 2.1079
step 4500: train loss 2.0419, val loss 2.1090
step 4750: train loss 2.0200, val loss 2.0886
step 5000: train loss 2.0183, val loss 2.0807

has wito me diss quir
St?

FRHALUS:
Everenle now yout Lorear that of mimser sine,
My of rettent deavessyince, yevatt-man, you'sfood unker greap do his fit' dis stry of thy aday to in the kand.

ISABOLINGLOUS:
And to marr, to I the brook chy emper,
That nable.

KING OLAU:
Hand
wer a scose prave ter do pirt mut this spak with thing is my of any ind be worn
Fales tisen.

KING sleawn: Vent 'teretech Elmo stes
it bet eyatescan to would froul she more his mep lettecer winch sels.

Sevice the IV:
That Lekes goodon all as anstishall of
Ands to wifuss a of I'll of gogne.

Senchand's faty and loxtry leet liberlcil,
Hand tas hey have it, his sway
Ju wat yought
Hall as preveral am you good meseeceir kall emp your thire somer damed
Wefiunfaie eyett wind dirst as breiciounc, nosiffuce.

KING I the a nese sudion bear, he coune.

First slaudre with our withn:
That himpell to sume, sparts of yher unlech
To no bow my modextion;
A ank tway light! I since has good race are and we to,
And trath
Cen me your


### Adding linear projection after attention

step 0: train loss 4.3591, val loss 4.3698
step 250: train loss 2.6070, val loss 2.6234
step 500: train loss 2.4530, val loss 2.4522
step 750: train loss 2.3446, val loss 2.3422
step 1000: train loss 2.2624, val loss 2.2899
step 1250: train loss 2.2284, val loss 2.2542
step 1500: train loss 2.2031, val loss 2.2310
step 1750: train loss 2.1717, val loss 2.2031
step 2000: train loss 2.1600, val loss 2.1991
step 2250: train loss 2.1522, val loss 2.1821
step 2500: train loss 2.1254, val loss 2.1712
step 2750: train loss 2.1153, val loss 2.1595
step 3000: train loss 2.0963, val loss 2.1643
step 3250: train loss 2.0797, val loss 2.1405
step 3500: train loss 2.0695, val loss 2.1196
step 3750: train loss 2.0527, val loss 2.1151
step 4000: train loss 2.0650, val loss 2.1274
step 4250: train loss 2.0393, val loss 2.1120
step 4500: train loss 2.0397, val loss 2.0994
step 4750: train loss 2.0242, val loss 2.0928
step 5000: train loss 2.0200, val loss 2.1036

Tailrmower sose thou fantage paacionds.

COLARD II:
Clay; we know;
A lutbadd you
Me theil as to hour on
Andsperifes. has tim.

PNUTENRETENl
My awn it will hap thous I'll stero-with apsold.

CUpay I pattellow in are to'd kinghtand what oft wher sole a dousioum to bre shy t our yeving mind lored
thou cont
may
There ele, and no if rive manteer prond cost flaigech hat, appland!
Gay of hell opcindy to O'll we the sprit:
Port to house have a, what is loge extrais,
Why way yenpitess age owhild chile to our more may the groce, a preccoll? senaver the grightand to my madies neavion.
But I wour Make have trucham at may havity, 'two, tuto thee is Covend
That wither will a reasue wand come clood but follier, and dear toe a those we gathers. Poorturs stay, our, from
Yot as at Ed Thom's you rehold mun peasterst's me gition be cliam on no cure
Freager
Eved atain
Seccher,
If.'s aneem is mt, his gat'st I here all bread till.
Ant 'tay, lews moke tore trrace be brost, are not ind.
What Hen, noost be I of