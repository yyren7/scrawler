## Page 1

# 2021 年度 前期課程入学試験問題 2 次募集 

江澤樹

Abstract. 2021 年 2 月 6,7 日に行われた名古屋大学大学院多元数理科学研究科博士前期課程の入試問題・解答・解説である.

1. 序

1 日目について, 1 において全体的に抽象度は低く基本的な計算が要求されている。(1) は固有値, 固有ベクトルの定義に従って成分計算をした。(2),(3) は Mathematica によって $A$ の固有多項式も計算し、検証した。(4) は多少は厄介に思えた。行列が対角化可能であるための必要十分条件が各固有値についてその重複度と固有空間の次元が等しい ということに注意している。また，解いているときに常に（1），（2），（3）の結果が利用できないかを意識した。方針として は（2）で $A$ は固有値 1 をもつことが示されているため，この固有値 1 の重複度に注目している。このとき（3）における $b=1$ という仮定にも注意している。 2 において各問は独立であることに注意した。（2）において Hessian が 0 になる ときは極値をとっていないと予想し, $(0,0)$ の近傍での挙動をを観察した。（3）は被積分関数に $x^{2}+y^{2}$ という式があっ たため極座標変換を行った。 3 において頻出問題であるが、積分経路が与えられていない。（1）は方程式の解を求める過程も詳しく書き、位数の説明も書いた。なお，偶然に因数分解により解ける形でもあった。（2）は積分経路の説明，不等式評価を詳しく書いた。問題文に「留数定理を用いて」とあるため部分分数分解による計算や $\Gamma$ 関数と $B$ 関数を用い た計算は別解とはできなかった。

2日目について, 1 において全体的に抽象度は高くない。（1）は問題文の誘導に従った。（2）は Hermite 行列の異な る固有値についての固有ベクトルは直交することの証明を思い出した。（3）を解く前に（4）の例を勘で考え，（3）の固有値についての結果を予想した。ユニタリ行列による対角化可能性については正規性 $A^{t} A={ }^{t} A A$ からも従う。 2 に おいて有界変動性が題材となっているのは珍しい。（1）は有名な事実であり平均値の定理から $f$ が Lipschitz 連続であ り Lipschitz 定数がどのように取れるのかを示した。（2）は（i）を解くにあたって最初から（1）の結果が使えないか考え た。（ii）では問題の流れからして有界変動でないと予想して考えた。 3 において（1），（2）は独立と思って解いた。（1）は Euclid 空間上の微分形式の線積分を勉強したときのことを思い出した。

## Contents

1 図 1
2 1 日目
3 2 日目
図類 11
図履履 11
図履履

---


## Page 2

# 2. I日目 

## 1 $a, b$ を実数とする. 3 次正方行列

$$
A=\left(\begin{array}{ccc}
-b & b+1 & a \\
-2 b & 2 b+1 & a \\
-b-1 & b+1 & a+1
\end{array}\right)
$$

について、以下の問に答えよ.
(1) $A$ の固有値で, $\binom{1}{1}$ という形の固有ベクトルをもつものをすべて求めよ.
(2) $A$ の固有値をすべて求めよ.
(3) $b=1$ の場合に, 固有値 1 に対する固有空間の次元を求めよ.
(4) $A$ が対角化不可能であるために $a, b$ がみたすべき必要十分条件を求めよ.

Proof. (1) 問題の固有値を $\lambda$ とする. また, 固有値 $\lambda$ に対する固有空間を $V_{\lambda}$ とおく、固有値と固有ベクトルの定義 III p.131, 第 5 章 $\S 5]$ より

$$
\left(\begin{array}{ccc}
-b & b+1 & a \\
-2 b & 2 b+1 & a \\
-b-1 & b+1 & a+1
\end{array}\right)\left(\begin{array}{l}
1 \\
1 \\
t
\end{array}\right)=\lambda\left(\begin{array}{l}
1 \\
1 \\
t
\end{array}\right)
$$

で, これは簡単に

$$
\left(\begin{array}{c}
t a+1 \\
t a+1 \\
t(a+1)
\end{array}\right)=\left(\begin{array}{c}
\lambda \\
\lambda \\
\lambda t
\end{array}\right)
$$

と計算できる. これをみたす $\lambda, t$ として

$$
(\lambda, t)=(1,0), \quad(a+1,1)
$$

が存在する.つまり, 求める固有値は $1, a+1$ であり,

$$
\left(\begin{array}{l}
1 \\
1 \\
0
\end{array}\right) \in V_{1}, \quad\left(\begin{array}{l}
1 \\
1 \\
1
\end{array}\right) \in V_{a+1} \quad(\forall a, b \in \mathbb{R})
$$

である.なお,この 2 つの固有ベクトルは $a$ の値に依存せずに線形独立である.とくに $a=0$ の場合は $A$ の固有値 1 の重複度は 2 以上であり, $\operatorname{dim} V_{1} \geq 2$ であることがわかる. このことは（4）において用いる。
(2)（1）より $A$ は $1, a+1$ を固有値にもつ. 残り 1 つを $\lambda$ とおく. $\operatorname{tr} A=-b+(2 b+1)+(a+1)$ であり, 一般に, 映 （trace）は固有値の和に等しい $\mathbb{R}$ p.146, 性質 IV]. つまり, $\operatorname{tr} A=1+(a+1)+\lambda$ でもある. よって, $\lambda=b$ である. よっ て, $A$ の固有値は

$$
1, \quad a+1, \quad b
$$

と求まる。
(3) $b=1$ の場合,

$$
A=\left(\begin{array}{ccc}
-1 & 2 & a \\
-2 & 3 & a \\
-2 & 2 & a+1
\end{array}\right)
$$

であり、行基本変形により

$$
I-A=\left(\begin{array}{ccc}
2 & -2 & -a \\
2 & -2 & -a \\
2 & -2 & -a
\end{array}\right) \rightarrow\left(\begin{array}{ccc}
1 & -1 & -a / 2 \\
0 & 0 & 0 \\
0 & 0 & 0
\end{array}\right)
$$

とできる.よって固有空間の定義と次元定理 III pp.108-109, 定理 4.5] より

$$
\operatorname{dim} V_{1}=\operatorname{dim} \operatorname{Ker}(I-A)=3-\operatorname{rank}(I-A)=3-1=2
$$

と求まる。
（4）結論： $A$ が対角化不可能であるために $a, b$ がみたすべき必要十分条件は $\lceil(a, b)=(0,1)$ または $a+1=b \neq \pm 1\rfloor$ である。理由：一般に，正方行列が対角化可能であるための必要十分条件は「各固有値の（固有多項式の根としての）重複度と固有空間の次元が一致している」ことである III p.136]. (2) より $A$ の固有値は $1, a+1, b$ である. これらの

---


## Page 3

重複とそのときの固有空間の次元を(i) $b=1$ の場合, (ii) $b \neq 1$ かつ $a+1=1$ の場合, (iii) $b \neq 1$ かつ $a+1 \neq 1$ の場合の 3 つの場合にわけて調べる。まず, (i) $b=1$ の場合ついて考える。このとき固有値 1 の重複度は 2 以上であり, (3) より $a$ によらずに $\operatorname{dim} V_{1}=2$ である。よって $A$ が対角化不可能であるためには固有値 1 の重複度が 3 となることが必要十分であるから $a=0$ である。つまり $(a, b)=(0,1)$ のとき $A$ は対角化不可能であることがわかる。次に, (ii) $b \neq 1$ かつ $a+1=1$ の場合について考える。このとき固有値 1 の重複度は 2 である。(1) の結果より $\operatorname{dim} V_{1} \geq 2$ であるため $A$ は対角化可能である。最後に, (iii) $b \neq 1$ かつ $a+1 \neq 1$ の場合について考える。 $A$ が対角不可能であるためには重複度が 2 以上の固有値の存在が必要であるから, $a+1=b$ としてよい。このとき

$$
A=\left(\begin{array}{ccc}
-b & b+1 & b-1 \\
-2 b & 2 b+1 & b-1 \\
-b-1 & b+1 & b
\end{array}\right), \quad b I-A=\left(\begin{array}{ccc}
2 b & -b-1 & -b+1 \\
2 b & -b-1 & -b+1 \\
b+1 & -b-1 & 0
\end{array}\right)
$$

である.簡単な小行列式の計算により, $a+1=b$ のもとでの $b I-A$ の階数は

$$
\operatorname{rank}(b I-A)=\left\{\begin{array}{cc}
1 & (b= \pm 1) \\
2 & (b \neq \pm 1)
\end{array}\right.
$$

であることが確認できるから，固有空間の定義と次元定理より

$$
\operatorname{dim} V_{a+1}=\operatorname{dim} V_{b}=\operatorname{dim} \operatorname{Ker}(b I-A)= \begin{cases}2 & (b= \pm 1) \\ 1 & (b \neq \pm 1)\end{cases}
$$

となり $a+1=b \neq \pm 1$ のとき $A$ は対角化不可能である。以上 (i), (ii), (iii) より求める条件は「 $(a, b)=(0,1)$ または $a+1=b \neq \pm 1$ 」である。

---


## Page 4

2 以下の問に答えよ.
(1) 広義積分

$$
\int_{1}^{\infty} \frac{\cos x}{x} d x
$$

が収束することを示せ.
(2) $\mathbb{R}^{2}$ 上で定義された 2 変数関数 $f(x, y)=x^{2}-y^{4}-(x+y)^{4}$ の極値をすべて求めよ.
(3) $a>0$ とする。次の積分値を求めよ.

$$
\int_{0}^{a}\left(\int_{y}^{\sqrt{2 a^{2}-y^{2}}} e^{x^{2}+y^{2}} d x\right) d y
$$

Proof. (1) 部分積分により任意の二つの正実数 $s<t$ に対して
$\left|\int_{s}^{t} \frac{\cos x}{x} d x\right|=\left|\left[\frac{\sin x}{x}\right]_{s}^{t}-\int_{s}^{t} \frac{\sin x}{-x^{2}} d x\right| \leq \frac{|\sin t|}{t}+\frac{|\sin s|}{s}+\int_{s}^{t} \frac{|\sin x|}{x^{2}} d x \leq \frac{1}{t}+\frac{1}{s}+\int_{s}^{t} \frac{1}{x^{2}} d x=\frac{1}{t}+\frac{1}{s}-\frac{1}{t}+\frac{1}{s}=\frac{2}{s}$
が得られる。よって、与えられた正実数 $\varepsilon$ に対し $c$ を $2 / \varepsilon$ より大きくとれば， $c<s<t$ を満たす任意の二つの実数 $s, t$ に対して

$$
\left|\int_{s}^{t} \frac{\cos x}{x} d x\right| \leq \frac{2}{s}<\varepsilon
$$

が成り立つことがわかる。このように Cauchy の判定法 猛 p.292,命題 11.1] の条件を満たすことが確かめられたため, この広義積分は収束する。
(2) 関数 $f$ の偏導関数は

$$
\frac{\partial f}{\partial x}(x, y)=2 x-4(x+y)^{3}, \quad \frac{\partial f}{\partial y}(x, y)=-4 y^{3}-4(x+y)^{3}
$$

であり,

$$
\frac{\partial f}{\partial x}(x, y)=\frac{\partial f}{\partial y}(x, y)=0 \Longleftrightarrow 2 x-4(x+y)^{3}=-4 y^{3}-4(x+y)^{3}=0
$$

なので停留点は

$$
(x, y)=(0,0), \quad( \pm 2, \mp 1)
$$

である（複号同順）。次にこの 3 つの点での Hessian を計算しよう。Hesse 行列は

$$
H_{f}(x, y):=\left(\begin{array}{cc}
\frac{\partial^{2} f}{\partial x^{2}}(x, y) & \frac{\partial^{2} f}{\partial x \partial y}(x, y) \\
\frac{\partial^{2} f}{\partial y \partial x}(x, y) & \frac{\partial^{2} f}{\partial y^{2}}(x, y)
\end{array}\right)=\left(\begin{array}{cc}
2-12(x+y)^{2} & -12(x+y)^{2} \\
-12(x+y)^{2} & -12 y^{2}-12(x+y)^{2}
\end{array}\right)
$$

より

$$
\operatorname{det} H_{f}(0,0)=\left|\begin{array}{ll}
2 & 0 \\
0 & 0
\end{array}\right|=0
$$

であり, Hessian からは何もわからない。しかし

$$
f(x, 0)=x^{2}-x^{4}>0 \quad(0<|x|<1), \quad f(0, y)=-2 y^{4}<0 \quad(y \neq 0)
$$

であるから $(x, y)=(0,0)$ で極値をとらない。

$$
\frac{\partial^{2} f}{\partial x^{2}}( \pm 2, \mp 1)=-10<0, \quad \operatorname{det} H_{f}( \pm 2, \mp 1)=\left|\begin{array}{cc}
-10 & -12 \\
-12 & -24
\end{array}\right|=96>0
$$

ゆえ $(x, y)=( \pm 2, \mp 1)$ で極大値をとる（複号同順）。以上を表にまとめて

| 停留点 | $f_{x x}$ | Hessian | 極値か? |
| :--: | :--: | :--: | :--: |
| $(0,0)$ | 2 | 0 | 極値をとらない |
| $(2,-1)$ | $-10(-)$ | $96(+)$ | 極大値 2 |
| $(-2,1)$ | $-10(-)$ | $96(+)$ | 極大値 2 |

から $f$ の極値は極大値 $f( \pm 2, \mp 1)=2$ のみである（複号同順）。

---


## Page 5

(3) $D:=\left\{(x, y) ; x^{2}+y^{2} \leq 2 a^{2}, 0 \leq y \leq x\right\}$ とおく、図示すると次のようになる：


極座標変換 $x=r \cos \theta, y=r \sin \theta$ により $D$ は $\Omega:=\{(r, \theta) ; 0 \leq r \leq \sqrt{2} a, 0 \leq \theta \leq \pi / 4\}$ に対応し、このときの Jacobian は $\frac{\partial(x, y)}{\partial(r, \theta)}=r$ である。よって

$$
\int_{0}^{a}\left(\int_{y}^{\sqrt{2 a^{2}-y^{2}}} e^{x^{2}+y^{2}} d x\right) d y=\iint_{D} e^{x^{2}+y^{2}} d x d y=\iint_{\Omega} e^{r^{2}} r d r d \theta=\int_{0}^{\pi / 4} d \theta \int_{0}^{\sqrt{2} a} e^{r^{2}} r d r=\frac{\pi}{8}\left(e^{2 a^{2}}-1\right)
$$

と求まる。

---


## Page 6

3. 広義積分

$$
I=\int_{-\infty}^{\infty} \frac{1}{x^{6}+1} d x
$$

を考える。以下の問に答えよ.
(1) 複素関数 $f(z)=\frac{1}{z^{6}+1}$ の上半平面 $\{z=x+i y \in \mathbb{C} ; x, y \in \mathbb{R}, y>0\}$ 内の極とそこでの留数を求めよ.
(2) 留数定理を用いて $I$ の値を求めよ.

Proof. (1) $P(z):=z^{6}+1, Q(z):=1$ とおくと， $f(z)=\frac{Q(z)}{P(z)}$ である。方程式 $P(z)=0$ の解を求めるため $z=r e^{i \theta}(r>$ $0, \theta \in \mathbb{R})$ とおき, $-1=e^{\pi i}$ であることに注意すると

$$
\begin{aligned}
z^{6}+1=0 & \Longleftrightarrow z^{6}=-1 \\
& \Longleftrightarrow\left(r e^{i \theta}\right)^{6}=e^{\pi i} \\
& \Longleftrightarrow r^{6} e^{6 i \theta}=e^{\pi i} \\
& \Longleftrightarrow r^{6}=1 \& e^{6 i \theta}=e^{\pi i} \\
& \Longleftrightarrow r=1 \& \exists n \in \mathbb{Z} \text { s.t. } 6 \theta=\pi+2 n \pi \\
& \Longleftrightarrow r=1 \& \exists n \in \mathbb{Z} \text { s.t. } \theta=\frac{\pi+2 n \pi}{6} \\
& \Longleftrightarrow \exists n \in \mathbb{Z} \text { s.t. } z=\exp \left(\frac{\pi+2 n \pi}{6} i\right)
\end{aligned}
$$

とできる。最後の式は $n$ について周期 6 であるので，連続する 6 個取ればすべてのものをつくす。よって $P(z)$ の零点 $c$ は

$$
c=e^{\frac{c}{6} i}, e^{\frac{c}{2} i}, e^{\frac{7 \pi}{6} i}, e^{\frac{7 \pi}{6} i}, e^{\frac{11 \pi}{6} i}
$$

つまり

$$
c= \pm i, \frac{ \pm \sqrt{3} \pm i}{2}
$$

である（復号任意）。これらの位数はみな 1 であり, $P, Q$ は $\mathbb{C}$ 全体で正則で， $Q(c)=1 \neq 0$ であるから, これらは $f$ の 1 位の極であり,

$$
\operatorname{Res}(f ; c)=\frac{Q(c)}{P^{\prime}(c)}=\frac{1}{6 c^{5}}=\frac{c}{6 c^{6}}=\frac{c}{6 \cdot(-1)}=-\frac{c}{6}
$$

である. 以上により極 $c$ のうちで上半平面にあるものは

$$
i, \quad \frac{ \pm \sqrt{3}+i}{2}
$$

でありそこでの留数は

$$
\operatorname{Res}(f ; i)=-\frac{i}{6}, \quad \operatorname{Res}\left(f ; \frac{ \pm \sqrt{3}+i}{2}\right)=-\frac{ \pm \sqrt{3}+i}{12}
$$

である（復号同順）。
【別解】代数的に解くには, $z^{6}=\left(z^{2}\right)^{3}$ に注意して因数分解することによって
$z^{6}+1=\left(z^{2}+1\right)\left(z^{4}-z^{2}+1\right)=\left(z^{2}+1\right)\left(z^{4}+2 z^{2}+1-3 z^{2}\right)=\left(z^{2}+1\right)\left\{\left(z^{2}+1\right)^{2}-(\sqrt{3} z)^{2}\right\}=\left(z^{2}+1\right)\left(z^{2}+\sqrt{3} z+1\right)\left(z^{2}-\sqrt{3} z+1\right)$
から， 3 つの 2 次方程式を解いて上の結果を得る。もしくは少し面倒で，気づきにくいかもしれないが $(-i)^{3}=i$ なので

$$
z^{6}+1=\left(z^{3}+i\right)\left(z^{3}-i\right)=(z+i)\left(z^{2}-i z+1\right)(z-i)\left(z^{2}+i z+1\right)
$$

ともできる。
(2) $\operatorname{deg} P(z)=6, \operatorname{deg} Q(z)=0, \operatorname{deg} P(z) \geq \operatorname{deg} Q(z)+2, \forall x \in \mathbb{R}, P(x) \neq 0$ であることに注意する。 $R \gg 1$ に対し て，実軸上の線分 $J_{R}:=\{z=x \in \mathbb{R} ;-R \leq x \leq R\}$ と上半円 $C_{R}:=\left\{z=R e^{i \theta} \in \mathbb{C} ; 0 \leq \theta \leq \pi\right\}$ からなる閉曲線に

---


## Page 7

反時計回りの向きを入れた積分経路を $\gamma_{R}:=J_{R}+C_{R}$ とおく。この閉曲線 $\gamma_{R}$ を図示すると


となり、留数定理[II. p.162, 定理 17] と（1）の結果より

$$
\begin{aligned}
\int_{\gamma_{R}} f(z) d z & =2 \pi i \sum_{\operatorname{Im} c>0} \operatorname{Res}(f ; c) \\
& \stackrel{(1)}{=} 2 \pi i\left\{\operatorname{Res}\left(f ; \frac{\sqrt{3}+i}{2}\right)+\operatorname{Res}(f ; i)+\operatorname{Res}\left(f ; \frac{-\sqrt{3}+i}{2}\right)\right\} \\
& \stackrel{(1)}{=} 2 \pi i\left(-\frac{\sqrt{3}+i}{12}-\frac{i}{6}-\frac{-\sqrt{3}+i}{12}\right) \\
& =\frac{2}{3} \pi
\end{aligned}
$$

である. ここで， $|z|=R$ のとき $\left|z^{6}+1\right| \geq|z|^{6}-1=R^{6}-1>0$ が成り立ち, $C_{R}$ の長さは $\pi R$ であるから

$$
\left|\int_{C_{R}} f(z) d z\right| \leq \frac{\pi R}{R^{6}-1} \rightarrow 0 \quad(R \rightarrow \infty)
$$

である. よって

$$
I=\int_{-\infty}^{\infty} f(x) d x=\lim _{R \rightarrow \infty} \int_{J_{R}} f(z) d z=\lim _{R \rightarrow \infty}\left(\int_{\gamma_{R}} f(z) d z-\int_{C_{R}} f(z) d z\right)=\frac{2}{3} \pi
$$

と求まる [III. p. 82 問題 3.5].
【補足】（2）で求めた積分 $I$ は留数定理を用いずに求めることもできる。被積分関数は有理関数なので部分分数分解 [II. p. 241 命題 6.1] を用いれば原理的には不定積分を計算できる [II. p.242, 命題 6.2]. 実際, (1) の【別解】の中で紹介した因数分解により部分分数分解は

$$
\frac{1}{x^{6}+1}=\frac{1}{3\left(x^{2}+1\right)}+\frac{\sqrt{3} x+2}{6\left(x^{2}+\sqrt{3} x+1\right)}+\frac{-\sqrt{3} x+2}{6\left(x^{2}-\sqrt{3} x+1\right)}
$$

であり, Mathematica によれば

$$
\int \frac{1}{x^{6}+1} d x=\frac{1}{4 \sqrt{3}} \log \frac{x^{2}+\sqrt{3} x+1}{x^{2}-\sqrt{3} x+1}+\frac{1}{6} \arctan (2 x+\sqrt{3})+\frac{1}{6} \arctan (2 x-\sqrt{3})+\frac{1}{3} \arctan x
$$

である。
【補足】 $\Gamma$ 関数と $B$ 関数とその性質，公式を用いる。ここに $\Gamma$ 関数と $B$ 関数の定義は

$$
\Gamma(x):=\int_{0}^{\infty} t^{x-1} e^{-t} d t, \quad B(x, y):=\int_{0}^{1} t^{x-1}(1-t)^{y-1} d t
$$

であり [II. p296, 定義 1], $B$ 関数について定義式の積分を変数変換することにより

$$
B(x, y)=\int_{0}^{\infty} \frac{t^{x-1}}{(1+t)^{x+y}} d t \quad\left(t \mapsto \frac{t}{1+t}\right)
$$

---


## Page 8

が成り立つ。また，相反公式 䏣 p.334，定理15.5］，「関数と $B$ 関数の関係 䏣 p.297，定理12.3］

$$
\Gamma(x) \Gamma(1-x)=\frac{\pi}{\sin \pi x}(0<x<1), \quad B(x, y)=\frac{\Gamma(x) \Gamma(y)}{\Gamma(x+y)}
$$

が成り立つ。次のように $x=t^{1 / 6}$ と置換すると

$$
\int_{0}^{\infty} \frac{1}{x^{6}+1} d x=\frac{1}{6} \int_{0}^{\infty} \frac{t^{-5 / 6}}{t+1} d t=\frac{1}{6} B\left(\frac{1}{6}, \frac{5}{6}\right)
$$

となるが、ここで

$$
B\left(\frac{1}{6}, \frac{5}{6}\right)=\frac{\Gamma\left(\frac{1}{6}\right) \Gamma\left(\frac{5}{6}\right)}{\Gamma\left(\frac{1}{6}+\frac{5}{6}\right)}=\Gamma\left(\frac{1}{6}\right) \Gamma\left(1-\frac{1}{6}\right)=\frac{\pi}{\sin \frac{\pi}{6}}=2 \pi
$$

より

$$
\int_{-\infty}^{\infty} \frac{1}{x^{6}+1} d x=2 \int_{0}^{\infty} \frac{1}{x^{6}+1} d x=2 \cdot \frac{1}{6} \cdot 2 \pi=\frac{2}{3} \pi
$$

と求まる。

---


## Page 9

# 3. 2日目 

$$
\begin{aligned}
& \text { (1) } n \text { 次実正方行列 } A \text { は } A^{2}={ }^{t} A \text { をみたすとする. ただし, }{ }^{t} A \text { は } A \text { の転置行列とする. } \mathbb{C}^{n} \text { の元 } x=\left(\begin{array}{c}
x_{1} \\
\vdots \\
x_{n}
\end{array}\right), y= \\
& \left(\begin{array}{c}
y_{1} \\
\vdots \\
y_{n}
\end{array}\right) \text { に対して, }\langle x, y\rangle \text { は標準 Hermite 内積 } \sum_{j=1}^{n} x_{j} \overline{y_{j}} \text { を表すものとする.以下の問に答えよ. }
\end{aligned}
$$

(1) $v$ を固有値 $\alpha$ に対する $A$ の固有ベクトルとする. $\langle v, A v\rangle$ を考察することにより $\alpha^{2}=\bar{\alpha}$ を示せ.
(2) 異なる固有値に対する $A$ の固有ベクトルは直交することを示せ.

以下, $n=3$ とし, $A$ は条件

$$
{ }^{T} A^{2}={ }^{t} A \text { が成り立ち,正則であり,かつ単位行列でない」 }
$$

をみたすとする。
(3) $A$ の固有値のうち少なくとも一つは実数であることを示し，さらに $A$ の固有値をすべて求めよ. また， $A$ はユニタリ行列により対角化されることを示せ.
(4) 条件をみたす $A$ の例を与えよ.

Proof. (1) 標準 Hermite 内積の性質 [III, pp.61-63] により

$$
\langle v, A v\rangle=\left\langle A^{*} v, v\right\rangle=\left\langle^{t} A v, v\right\rangle=\left\langle A^{2} v, v\right\rangle=\left\langle\alpha^{2} v, v\right\rangle=\alpha^{2}\langle v, v\rangle
$$

である. ただし， $A^{*}$ は $A$ の随伴行列である。一方で，標準 Hermite 内積の第二変数に関する共役線形性により

$$
\langle v, A v\rangle=\langle v, \alpha v\rangle=\bar{\alpha}\langle v, v\rangle
$$

である. よって, 固有ベクトル $v$ の任意性により $\alpha^{2}=\bar{\alpha}$ である。
(2) $v, w$ をそれぞれ固有値 $\alpha, \beta$ に対する $A$ の固有ベクトルとする. 標準 Hermite 内積の第一変数に関する線形性に より

$$
\langle A v, w\rangle=\langle\alpha v, w\rangle=\alpha\langle v, w\rangle
$$

である. 一方, 標準 Hermite 内積の第二変数に関する共役線形性と（1）の結果を用いると

$$
\langle A v, w\rangle=\left\langle v,{ }^{t} A w\right\rangle=\left\langle v, A^{2} w\right\rangle=\left\langle v, \beta^{2} w\right\rangle=\overline{\beta^{2}}\langle v, w\rangle \stackrel{(1)}{=} \overline{\beta}\langle v, w\rangle=\beta\langle v, w\rangle
$$

である. よって $\alpha \neq \beta$ であれば $\langle v, w\rangle=0$ でなければならない。よって異なる固有値に対する $A$ の固有ベクトルは直交する。
(3) $A$ は 3 次実正方行列であるからその固有多項式 $|x I-A|$ は実数係数 3 次多項式である. よって中間値の定理 $\mathbb{R}$ p.74, 定理 8.1] よりその根のうち少なくとも一つは実数である。つまり $A$ の固有値のうち少なくとも一つは実数であ る. また，（1）より $A$ の実固有値は 0,1 のどちらかである。条件より $A$ は正則であるため 0 を固有値にもたない。つまり $A$ は固有値 1 をもち, 1 以外に実固有値はもたない。条件より $A$ は単位行列ではないから実数でない固有値 $\lambda$ ももつこ とになる。 $A$ は実行列であったから $\bar{\lambda}$ もまた $A$ の固有値である。（1）の結果と de Moivre の定理から $\lambda$ の絶対値，偏角 が計算できて, $A$ の固有値は

$$
1, \frac{-1 \pm \sqrt{3} i}{2}
$$

の三つであることがわかる。 これと（2）より各固有値に対する固有ベクトルは直交するから，それぞれ正規化して並べ れば $A$ を対角化するユニタリ行列を構成できる [III, p.64].
(4) 置換行列で

$$
A:=\left(\begin{array}{lll}
0 & 1 & 0 \\
0 & 0 & 1 \\
1 & 0 & 0
\end{array}\right)
$$

とすればよい。これが条件をみたすことは容易に確認できる。

---


## Page 10

2. $I=[0,1]$ とする. $I$ 上で定義された関数 $\varphi$ は, ある $M>0$ が存在して $I$ の任意の分割

$$
0=t_{0}<t_{1}<\cdots<t_{n}=1
$$

に対して

$$
\sum_{j=0}^{n-1}\left|\varphi\left(t_{j+1}\right)-\varphi\left(t_{j}\right)\right| \leq M
$$

をみたすとき, $I$ で有界変動であるという.以下の問に答えよ.
(1) $f$ は $I$ を含む開区間で微分可能で，その導関数 $f^{\prime}$ は $I$ で有界とする。このとき $f$ は $I$ で有界変動である ことを示せ.
(2) 次の関数は $I$ で有界変動であるかどうか, 理由とともに答えよ.

$$
\text { (i) } g(x)=\left\{\begin{array}{ll}
x^{2} \sin \frac{1}{x} & (x \neq 0) \\
0 & (x=0)
\end{array} \quad \text { (ii) } \quad h(x)=\left\{\begin{array}{ll}
x \cos \frac{1}{x} & (x \neq 0) \\
0 & (x=0)
\end{array}\right.\right.
$$

Proof. (1) 任意の分割 $0=t_{0}<t_{1}<\cdots<t_{n}=1$ に対して, 平均値の定理 $\mathbb{\&}$ p.93, 定理 2.3 より

$$
f\left(t_{j+1}\right)-f\left(t_{j}\right)=f^{\prime}\left(\tau_{j}\right)\left(t_{j+1}-t_{j}\right), \quad t_{j}<\tau_{j}<t_{j+1}
$$

とできる. ここで, $M:=\sup _{t \in I}\left|f^{\prime}(t)\right|$ とおいて

$$
\left|f\left(t_{j+1}\right)-f\left(t_{j}\right)\right| \leq M\left|t_{j+1}-t_{j}\right| \quad(\forall j=0,1, \ldots, n-1)
$$

から $j$ について和をとれば

$$
\sum_{j=0}^{n-1}\left|f\left(t_{j+1}\right)-f\left(t_{j}\right)\right| \leq M \sum_{j=0}^{n-1}\left|t_{j+1}-t_{j}\right|=M(1-0)=M
$$

とできるため $f$ は $I$ で有界変動である $\mathbb{\&}$ p.345, 定理 16.2 2)].
(2) (i) 有界変動である. 理由：まず，

$$
\frac{g(0+\varepsilon)-g(0)}{\varepsilon}=\frac{\varepsilon^{2} \sin \frac{1}{\varepsilon}-0}{\varepsilon}=\varepsilon \sin \frac{1}{\varepsilon} \rightarrow 0 \quad(\varepsilon \rightarrow 0)
$$

と積の微分公式から

$$
g^{\prime}(x)= \begin{cases}2 x \sin \frac{1}{x}-x \cos \frac{1}{x} & (x \neq 0) \\ 0 & (x=0)\end{cases}
$$

である. よって, 任意の $x \in I$ に対して

$$
\left|g^{\prime}(x)\right| \leq\left|2 x \sin \frac{1}{x}-x \cos \frac{1}{x}\right| \leq 2|x|\left|\sin \frac{1}{x}\right|+|x|\left|\cos \frac{1}{x}\right| \leq 2|x|+|x|=3|x| \leq 3
$$

ゆえ $g$ は $I$ を含む開区間で微分可能で，その導関数 $g^{\prime}$ は $I$ で有界である. よって（1）より $g$ は $I$ で有界変動である。（ii）有界変動でない。理由：いま $I$ の分割として

$$
0<\frac{2}{4 n \pi}<\frac{2}{(4 n-1) \pi}<\frac{2}{(4 n-2) \pi}<\cdots<\frac{2}{4 \pi}<\frac{2}{3 \pi}<\frac{2}{2 \pi}<1
$$

なる $4 n+1$ 個の分点より成るものを取れば

$$
h\left(\frac{2}{4 j \pi}\right)=\frac{2}{4 j \pi}, h\left(\frac{2}{(4 j-1) \pi}\right)=0, \quad h\left(\frac{2}{(4 j-2) \pi}\right)=-\frac{2}{(4 j-2) \pi}, h\left(\frac{2}{(4 j-3) \pi}\right)=0
$$

だからこの分割について, $h$ の $I$ における変動量 $\mathbb{\&}$ p.345, 定義 4] の下からの評価が

$$
2 \sum_{j=1}^{n}\left(\frac{2}{4 j \pi}+\frac{2}{(4 j-2) \pi}\right)=\frac{4}{\pi} \sum_{j=1}^{2 n} \frac{1}{j} \rightarrow+\infty \quad(n \rightarrow+\infty)
$$

となることによる $\mathbb{\&}$ p.343, 例 3].

---


## Page 11

3 以下の問に答えよ.
(1) $\mathbb{R}^{n}$ の開集合 $U(\neq \emptyset)$ 上の実数値関数 $f: U \rightarrow \mathbb{R}$ が $C^{1}$ 級で偏導関数 $\frac{\partial f}{\partial x_{1}}, \frac{\partial f}{\partial x_{2}}, \ldots, \frac{\partial f}{\partial x_{n}}$ が $U$ 上で恒等的に零であるとする。 $U$ が条件

「 $U$ の任意の 2 点は $U$ 内の $C^{1}$ 級曲線で結べる」
をみたすならば， $f$ は定数関数であることを示せ。
（2）位相空間 $X(\neq \emptyset)$ 上の実数値関数 $f: X \rightarrow \mathbb{R}$ が局所定数関数であるとは、任意の $p \in X$ に対して， $p$ を含む開集合 $V$ が存在して， $f$ が $V$ 上で定数であることとする。 $X$ が連結ならば， $X$ 上の局所定数関数は $X$ 上の定数関数であることを示せ.

Proof. (1) 任意の $p, q \in U$ を固定し, $p$ から $q$ への $C^{1}$ 級曲線 $\gamma:[a, b] \rightarrow U(\gamma(a)=p, \gamma(b)=q)$ をとる。仮定より $f, \gamma$ が $C^{1}$ 級であるから $f \circ \gamma$ も $C^{1}$ 級で，微分積分学の基本定理，合成関数の微分公式および問題文の条件から

$$
f(q)-f(p)=f(\gamma(b))-f(\gamma(a))=\int_{a}^{b} \frac{d(f \circ \gamma)}{d t}(t) d t=\int_{a}^{b}\left(\frac{\partial f}{\partial x_{1}}(\gamma(t)) \frac{d \gamma_{1}}{d t}(t)+\cdots+\frac{\partial f}{\partial x_{n}}(\gamma(t)) \frac{d \gamma_{n}}{d t}(t)\right) d t=0
$$

と計算できる。ここで， $\gamma(t)=\left(\gamma_{1}(t), \cdots, \gamma_{n}(t)\right)$ である。以上により $f$ は定数関数である $\square$ 図。
（2）1点部分集合の逆像 $f^{-1}(\{a\})$ は $X$ の開集合である。実際, $p \in f^{-1}(\{a\})$ とすれば $f(p)=a$ であるから $p$ を含 む開集合 $V_{p}$ が存在して， $f$ が $V_{p}$ 上で定数 $a$ である。このとき $V_{p} \subset f^{-1}(\{a\})$ ゆえ $\bigcup_{f(p)=a} V_{p}=f^{-1}(\{a\})$ であるから やはり $f^{-1}(\{a\})$ は $X$ の開集合である。さて， $a \neq b$ ならば $f^{-1}(\{a\}) \cap f^{-1}(\{b\})=\emptyset$ である。 $X=\bigcup_{a \in f(X)} f^{-1}(\{a\})$ であるから $f(X)$ が 2 元以上含めば $X$ が連結であることに反する。

# 謝辞 

全体的に河合将吾くんからの指摘により誤植などを補うことができた。ここに感謝します。

## REFERENCES

[1] 三宅敏恒, 入門微分積分, 培風館.
[2] 川平友規, 微分積分, 日本評論社.
[3] 笠原晴司, 微分積分学, サイエンス社.
[4] 杉浦光夫, 解析入門 I, 東京大学出版会.
[5] 杉浦光夫, 解析入門 II, 東京大学出版会.
[6] 伊藤清三, ルベーグ積分入門, 裳華房.
[7] 柴田良弘, ルベーグ積分論, 内田老鶴圃.
[8] 村上正康・佐藤恒雄・野澤宗平・稲葉尚志, 教養の線形代数, 培風館.
[9] 上坂吉則・塚田真, 入門線形代数, 近代科学社.
[10] 斎藤正彦, 線型代数入門, 東京大学出版会.
[11] 佐武一郎, 線型代数学, 裳華房.
[12] 川平友規, 入門 複素関数,裳華房.
[13] 殿塚勲・河村哲也, 理工系の複素関数論, 東京大学出版会.
[14] 今吉洋一, 複素関数概説, サイエンス社.
[15] 梶原壊二, 関数論入門, 森北出版株式会社.
[16] L.V. Ahlfors, Complex Analysis, McGraw Hill.
[17] L.V. アールフォルス, 複素解析, 現代数学社.
[18] 坪井俊, 幾何学 III, 東京大学出版会.
[19] 松本幸夫, トポロジー入門, 岩波書店.
[20] 森田茂之, 集合と位相空間, 朝倉書店.
[21] 松坂和夫, 集合・位相入門, 岩波書店.
[22] 矢野公一, 新離空間と位相構造, 共立出版株式会社.
[23] 水田義弘, 詳解演習微分積分, サイエンス社.
[24] 水田義弘, 詳解演習線形代数, サイエンス社.
[25] 藤家龍雄・岸正倫, 関数論演習, サイエンス社.
[26] 日本数学会, 岩波 数学辞典 第 3 版, 岩波書店.
[27] 小池慎一, Mathematica 数式処理入門, 技術評論社.
[28] 奥村晴彦, $\mathrm{LAT}_{\mathrm{E}} \mathrm{X} 2 \mathrm{c}$ 実文書作成入門 改訂第 5 版, 技術評論社.
[29] https://www.math.nagoya-u.ac.jp/ja/admission/gs/exam.html

---

